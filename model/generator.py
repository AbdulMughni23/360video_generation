"""
model/generator.py — Full Generator for the Text-to-360° Video GAN.

Pipeline:
  Stage 1 — WanBackbone        (frozen, non-trainable) → [B, T, C, H=512, W=1024]
  Stage 2 — WrapLayer360       (fixed, non-trainable)  → [B, T, C, H=572, W=1124]
  Stage 3 — CNNRefinement      (TRAINABLE)             → [B, T, C, H=572, W=1124]
  Stage 4 — ERPProjection      (fixed, non-trainable)  → [B, T, C, H=512, W=1024]

Only Stage 3 parameters are optimised during training.
"""

import math
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    BACKBONE_LOCAL_DIR,
    BACKBONE_STEPS,
    BACKBONE_GUIDANCE,
    CNN_CHANNELS,
    DROPOUT_RATE,
    WRAP_PAD_H,
    WRAP_PAD_W,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Wan2.1-T2V-1.3B BACKBONE  (Frozen, Non-Trainable)
# ══════════════════════════════════════════════════════════════════════════════

class WanBackbone(nn.Module):
    """
    Wraps the Wan2.1-T2V-1.3B diffusion pipeline.
    ALL weights are permanently frozen — zero trainable parameters.
    Always runs inference on cuda:0 regardless of DataParallel assignment.
    """

    def __init__(self):
        super().__init__()

        # ── Load pipeline ───────────────────────────────────────────────────
        try:
            from diffusers import WanPipeline
            self.pipe = WanPipeline.from_pretrained(
                str(BACKBONE_LOCAL_DIR),
                torch_dtype=torch.float16,
            )
            logger.info("WanPipeline loaded from %s", BACKBONE_LOCAL_DIR)
        except (ImportError, AttributeError):
            logger.warning(
                "WanPipeline not available — falling back to "
                "AutoPipelineForText2Video"
            )
            from diffusers import AutoPipelineForText2Video
            self.pipe = AutoPipelineForText2Video.from_pretrained(
                str(BACKBONE_LOCAL_DIR),
                torch_dtype=torch.float16,
            )
            logger.info(
                "AutoPipelineForText2Video loaded from %s", BACKBONE_LOCAL_DIR
            )

        # Always live on cuda:0 — too large to replicate across GPUs
        self.pipe.to("cuda:0")

        # ── Freeze ALL backbone weights ─────────────────────────────────────
        for param in self.pipe.parameters():
            param.requires_grad_(False)

        # Verify: assert no backbone weights are trainable
        assert all(not p.requires_grad for p in self.pipe.parameters()), (
            "All Wan2.1 backbone weights must be frozen"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _extract_tensor(self, result) -> torch.Tensor:
        """
        Extract a [T=100, C=3, H=512, W=1024] float32 CPU tensor from any
        pipeline output format.

        Handles:
          Case A — result.frames is a tensor  [1,T,C,H,W] or [T,C,H,W]
          Case B — result.frames is a list of T PIL images
          Case C — result.frames is a numpy array [T,H,W,C]
        """
        import numpy as np
        from PIL import Image

        frames = result.frames

        # ── Case A: tensor ───────────────────────────────────────────────
        if isinstance(frames, torch.Tensor):
            tensor = frames
            if tensor.dim() == 5:           # [1, T, C, H, W]
                tensor = tensor.squeeze(0)  # → [T, C, H, W]
            # tensor is now [T, C, H, W]

        # ── Case B: list of PIL images ───────────────────────────────────
        elif isinstance(frames, (list, tuple)) and isinstance(
            frames[0], Image.Image
        ):
            np_frames = [np.array(img) for img in frames]  # list of [H,W,C]
            arr = np.stack(np_frames, axis=0)               # [T, H, W, C]
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T,C,H,W]

        # ── Case C: numpy array [T, H, W, C] ────────────────────────────
        else:
            import numpy as np
            arr = np.array(frames)  # ensure it is a proper ndarray
            if arr.ndim == 4 and arr.shape[-1] == 3:
                tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Unknown backbone output format: type={type(frames)}, "
                    f"shape={getattr(arr, 'shape', 'N/A')}"
                )

        # ── Normalise to float32 [-1, 1] ─────────────────────────────────
        tensor = tensor.float()
        max_val = tensor.max().item()
        min_val = tensor.min().item()

        if max_val > 1.5:
            # Values in [0, 255]
            tensor = tensor / 127.5 - 1.0
        elif min_val >= 0.0 and max_val <= 1.0:
            # Values in [0.0, 1.0]
            tensor = tensor * 2.0 - 1.0
        # else: already in [-1, 1], no change

        assert tensor.shape == (100, 3, 512, 1024), (
            f"Backbone output shape mismatch: {tensor.shape}"
        )

        return tensor.cpu()

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, prompts: List[str]) -> torch.Tensor:
        """
        Run Wan2.1 inference for each prompt individually to fit VRAM.

        Args:
            prompts: List of B text prompt strings.

        Returns:
            FloatTensor [B, T=100, C=3, H=512, W=1024], float32, [-1,1], CPU.
        """
        outputs = []
        for prompt in prompts:
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    height=512,
                    width=1024,
                    num_frames=100,
                    num_inference_steps=BACKBONE_STEPS,
                    guidance_scale=BACKBONE_GUIDANCE,
                    output_type="pt",
                )
            tensor = self._extract_tensor(result)  # [T=100, C=3, H=512, W=1024]
            outputs.append(tensor)

        x = torch.stack(outputs, dim=0)  # [B, T, C, H, W]
        return x


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — 360° WRAP LAYER  (Fixed, Non-Trainable)
# ══════════════════════════════════════════════════════════════════════════════

class WrapLayer360(nn.Module):
    """
    Applies circular (toroidal) padding to simulate the continuity of a 360°
    equirectangular projection.

    Input:  [B, T, C, H=512,  W=1024 ]
    Output: [B, T, C, H=572,  W=1124 ]
            H_out = 512 + 2 * WRAP_PAD_H = 572
            W_out = 1024 + 2 * WRAP_PAD_W = 1124

    Contains ZERO learnable parameters.
    """

    def __init__(self):
        super().__init__()
        # Stateless layer — padding constants come from config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H=512, W=1024]

        Returns:
            [B, T, C, H=572, W=1124]
        """
        # ── Horizontal circular padding (W axis, dim=4) ───────────────────
        left_pad  = x[:, :, :, :, -WRAP_PAD_W:]   # rightmost 50 cols → left
        right_pad = x[:, :, :, :,  :WRAP_PAD_W]   # leftmost  50 cols → right
        x = torch.cat([left_pad, x, right_pad], dim=4)
        # shape: [B, T, C, 512, 1124]

        # ── Vertical circular padding (H axis, dim=3) ─────────────────────
        top_pad    = x[:, :, :, -WRAP_PAD_H:, :]  # bottom 30 rows → top
        bottom_pad = x[:, :, :,  :WRAP_PAD_H, :]  # top    30 rows → bottom
        x = torch.cat([top_pad, x, bottom_pad], dim=3)
        # shape: [B, T, C, 572, 1124]

        return x


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CNN REFINEMENT BLOCKS  (Trainable)
# ══════════════════════════════════════════════════════════════════════════════

class CNNBlock(nn.Module):
    """
    Single CNN refinement block operating on [B, T, C_hidden, H, W].

    Contains two residual 3-D convolutions:
      1. Temporal conv  (3×1×1) — models motion across 3 consecutive frames.
      2. Spatial  conv  (1×3×3) — models local spatial patterns.

    Caller always provides and receives tensors in [B, T, C, H, W] layout.
    Internally we permute to [B, C, T, H, W] for Conv3d compatibility.
    """

    def __init__(self, channels: int, dropout_rate: float):
        super().__init__()

        # ── 1. Temporal Convolution ────────────────────────────────────────
        self.temporal_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            bias=False,
        )
        self.temporal_norm = nn.GroupNorm(
            num_groups=min(8, channels), num_channels=channels
        )
        self.temporal_act = nn.GELU()

        # ── 2. Spatial Convolution ─────────────────────────────────────────
        self.spatial_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            bias=False,
        )
        self.spatial_norm = nn.GroupNorm(
            num_groups=min(8, channels), num_channels=channels
        )
        self.spatial_act = nn.GELU()

        # ── 3. Dropout ─────────────────────────────────────────────────────
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]

        Returns:
            [B, T, C, H, W]
        """
        # Permute to [B, C, T, H, W] for Conv3d
        x = x.permute(0, 2, 1, 3, 4)   # [B, C, T, H, W]

        # ── Temporal residual block ────────────────────────────────────────
        residual = x
        x = self.temporal_conv(x)
        x = x + residual
        x = self.temporal_norm(x)
        x = self.temporal_act(x)

        # ── Spatial residual block ─────────────────────────────────────────
        residual = x
        x = self.spatial_conv(x)
        x = x + residual
        x = self.spatial_norm(x)
        x = self.spatial_act(x)

        # ── Dropout ────────────────────────────────────────────────────────
        x = self.dropout(x)

        # Permute back to [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)   # [B, T, C, H, W]
        return x


class CNNRefinement(nn.Module):
    """
    Three stacked CNNBlock layers with input and output projection convolutions.

    Input:  [B, T=100, C=3,          H=572, W=1124]
    Output: [B, T=100, C=3,          H=572, W=1124]

    THIS IS THE ONLY TRAINABLE MODULE IN THE ENTIRE GENERATOR.
    """

    def __init__(self, channels: int = CNN_CHANNELS,
                 dropout_rate: float = DROPOUT_RATE):
        super().__init__()

        # ── Input projection: 3 → channels ────────────────────────────────
        # We apply Conv3d so permute/permute-back is needed here too.
        self.input_proj_conv = nn.Conv3d(
            in_channels=3,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self.input_proj_norm = nn.GroupNorm(
            num_groups=8, num_channels=channels
        )
        self.input_proj_act = nn.GELU()

        # ── Three independent CNNBlocks ────────────────────────────────────
        self.block1 = CNNBlock(channels=channels, dropout_rate=dropout_rate)
        self.block2 = CNNBlock(channels=channels, dropout_rate=dropout_rate)
        self.block3 = CNNBlock(channels=channels, dropout_rate=dropout_rate)

        # ── Output projection: channels → 3 ───────────────────────────────
        self.output_proj_conv = nn.Conv3d(
            in_channels=channels,
            out_channels=3,
            kernel_size=1,
            bias=False,
        )
        self.output_act = nn.Tanh()  # clamp output to [-1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C=3, H=572, W=1124]

        Returns:
            [B, T, C=3, H=572, W=1124]
        """
        # ── Input projection ───────────────────────────────────────────────
        # Permute to [B, C, T, H, W] for Conv3d
        x = x.permute(0, 2, 1, 3, 4)           # [B, C, T, H, W]
        x = self.input_proj_conv(x)
        x = self.input_proj_norm(x)
        x = self.input_proj_act(x)
        x = x.permute(0, 2, 1, 3, 4)           # [B, T, C, H, W]

        # ── Three CNN blocks ───────────────────────────────────────────────
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # ── Output projection ──────────────────────────────────────────────
        x = x.permute(0, 2, 1, 3, 4)           # [B, C, T, H, W]
        x = self.output_proj_conv(x)
        x = x.permute(0, 2, 1, 3, 4)           # [B, T, C, H, W]
        x = self.output_act(x)                  # Tanh → [-1, 1]

        return x


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — ERP PROJECTION + CROP  (Fixed, Non-Trainable)
# ══════════════════════════════════════════════════════════════════════════════

class ERPProjection(nn.Module):
    """
    Crop the circular wrap padding and apply a latitude-based ERP spherical
    correction weight that suppresses the polar distortion inherent in
    equirectangular projection.

    Input:  [B, T=100, C=3, H=572,  W=1124]
    Output: [B, T=100, C=3, H=512,  W=1024]

    Contains ZERO learnable parameters.
    The ERP weight map is a registered buffer (auto-moves with .to(device)).
    """

    def __init__(self):
        super().__init__()

        # ── Precompute latitude-based ERP correction weight ────────────────
        H, W = 512, 1024
        # Latitude ranges from +π/2 (north pole) to -π/2 (south pole)
        lat    = torch.linspace(math.pi / 2, -math.pi / 2, H)  # [H]
        weight = torch.cos(lat)                                   # [H], ∈[0,1]
        weight = weight.view(1, 1, 1, H, 1)                      # broadcast

        # Register as buffer so it automatically moves with .to(device)
        self.register_buffer("erp_weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H=572, W=1124]

        Returns:
            [B, T, C, H=512, W=1024]
        """
        # ── Step A: Crop wrap padding ──────────────────────────────────────
        x = x[:, :, :,
               WRAP_PAD_H:-WRAP_PAD_H,
               WRAP_PAD_W:-WRAP_PAD_W]
        # shape: [B, T, C, 512, 1024]

        # ── Step B: ERP latitude correction ───────────────────────────────
        # erp_weight: [1, 1, 1, 512, 1] — broadcasts over B, T, C, W
        # Equator pixels (cos ≈ 1.0) unchanged
        # Pole pixels (cos ≈ 0.0) suppressed
        x = x * self.erp_weight

        # ── Step C: Clamp to [-1, 1] ───────────────────────────────────────
        x = torch.clamp(x, -1.0, 1.0)

        return x


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR — Assembles all 4 stages
# ══════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    Text-to-360° Video Generator.

    Input:  list of B text prompt strings
    Output: FloatTensor [B, T=100, C=3, H=512, W=1024], float32, [-1, 1]

    Only Stage 3 (CNNRefinement) is trainable. All other weights are frozen.
    """

    def __init__(self):
        super().__init__()

        self.backbone = WanBackbone()     # Stage 1 — frozen
        self.wrap     = WrapLayer360()    # Stage 2 — no params
        self.cnn      = CNNRefinement()   # Stage 3 — TRAINABLE
        self.erp      = ERPProjection()   # Stage 4 — no params

        # ── Verify that backbone sub-module has no trainable params ────────
        for param in self.backbone.parameters():
            assert not param.requires_grad, (
                "Backbone parameters must not be trainable inside Generator"
            )

        # ── Verify that ONLY Stage 3 (CNN) parameters are trainable ────────
        for name, param in self.named_parameters():
            if "cnn" not in name:
                assert not param.requires_grad, (
                    f"Non-CNN parameter '{name}' must not be trainable"
                )

        print("Generator check passed: only Stage 3 CNN blocks are trainable.")

    def forward(self, prompts: List[str]) -> torch.Tensor:
        """
        Full generator forward pass through all 4 stages.

        Args:
            prompts: List of B text prompt strings.

        Returns:
            FloatTensor [B, T=100, C=3, H=512, W=1024], float32, [-1, 1].
        """
        # Stage 1 — Wan2.1 backbone (returns CPU float32 tensor)
        x = self.backbone(prompts)         # [B, T, C, H=512,  W=1024]

        # Move to the device that the CNN (stage 3) lives on
        x = x.to(next(self.cnn.parameters()).device)

        # Stage 2 — 360° circular wrap padding
        x = self.wrap(x)                   # [B, T, C, H=572,  W=1124]

        # Stage 3 — CNN refinement (only trainable stage)
        x = self.cnn(x)                    # [B, T, C, H=572,  W=1124]

        # Stage 4 — ERP projection + crop + latitude correction
        x = self.erp(x)                    # [B, T, C, H=512,  W=1024]

        return x