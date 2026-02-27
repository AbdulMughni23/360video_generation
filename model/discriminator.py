"""
model/discriminator.py — Discriminator for the Text-to-360° Video GAN.

Unconditional video discriminator.
Outputs a raw Wasserstein score (no sigmoid).
Uses GroupNorm only — no BatchNorm anywhere.
All parameters are trainable.
"""

import torch
import torch.nn as nn
from torch import Tensor

from config import DROPOUT_RATE


class DiscBlock(nn.Module):
    """
    Single discriminator block: temporal conv then spatial-downsampling conv.

    Input:  [B, in_ch,  T, H, W]
    Output: [B, out_ch, T, H/2, W/2]

    Note: The block receives and returns tensors in [B, C, T, H, W] layout
    (i.e., the "conv-first" layout), unlike the generator which uses
    [B, T, C, H, W].  The parent Discriminator handles the outer permutations.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float):
        super().__init__()

        # ── 1. Temporal Conv (no spatial downsampling) ─────────────────────
        self.temporal_conv = nn.Conv3d(
            in_ch, in_ch,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            bias=False,
        )
        self.temporal_norm = nn.GroupNorm(
            num_groups=min(8, in_ch), num_channels=in_ch
        )
        self.temporal_act = nn.LeakyReLU(0.2, inplace=True)

        # ── 2. Spatial Conv (stride=2, halves H and W) ─────────────────────
        self.spatial_conv = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
        )
        self.spatial_norm = nn.GroupNorm(
            num_groups=min(8, out_ch), num_channels=out_ch
        )
        self.spatial_act = nn.LeakyReLU(0.2, inplace=True)

        # ── 3. Dropout ─────────────────────────────────────────────────────
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, in_ch, T, H, W]

        Returns:
            [B, out_ch, T, H/2, W/2]
        """
        # Temporal pass
        x = self.temporal_conv(x)
        x = self.temporal_norm(x)
        x = self.temporal_act(x)

        # Spatial downsampling pass
        x = self.spatial_conv(x)
        x = self.spatial_norm(x)
        x = self.spatial_act(x)

        # Dropout
        x = self.dropout(x)

        return x


class Discriminator(nn.Module):
    """
    Unconditional video discriminator for 360° equirectangular videos.

    Input:  FloatTensor [B, T=100, C=3, H=512, W=1024]  (real or fake)
    Output: FloatTensor [B, 1]  — raw Wasserstein score, NO sigmoid

    Architecture:
      Layer 0:  Input projection   [B,   3, T, 512, 1024] → [B,  64, T, 256, 512]
      Block 1:  DiscBlock 64→128   → [B, 128, T, 128, 256]
      Block 2:  DiscBlock 128→256  → [B, 256, T,  64, 128]
      Block 3:  DiscBlock 256→512  → [B, 512, T,  32,  64]
      Block 4:  DiscBlock 512→512  → [B, 512, T,  16,  32]
      Block 5:  DiscBlock 512→512  → [B, 512, T,   8,  16]
      Pool:     AdaptiveAvgPool3d(1,1,1) → [B, 512, 1, 1, 1]
      Flatten:  → [B, 512]
      FC:       512 → 128 → 1  (LeakyReLU between, no activation on output)
    """

    def __init__(self, dropout_rate: float = DROPOUT_RATE):
        super().__init__()

        # ── Layer 0: Input projection ──────────────────────────────────────
        # Spatial: 512×1024 → 256×512  (stride 2 on H and W)
        self.input_proj = nn.Sequential(
            nn.Conv3d(
                3, 64,
                kernel_size=(1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Five discriminator blocks ──────────────────────────────────────
        self.block1 = DiscBlock(in_ch=64,  out_ch=128, dropout_rate=dropout_rate)
        self.block2 = DiscBlock(in_ch=128, out_ch=256, dropout_rate=dropout_rate)
        self.block3 = DiscBlock(in_ch=256, out_ch=512, dropout_rate=dropout_rate)
        self.block4 = DiscBlock(in_ch=512, out_ch=512, dropout_rate=dropout_rate)
        self.block5 = DiscBlock(in_ch=512, out_ch=512, dropout_rate=dropout_rate)

        # ── Global pooling ─────────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        # ── Output head ────────────────────────────────────────────────────
        # Raw Wasserstein score — NO sigmoid
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: FloatTensor [B, T=100, C=3, H=512, W=1024]

        Returns:
            FloatTensor [B, 1] — raw Wasserstein score (no sigmoid)
        """
        # ── Permute to [B, C, T, H, W] for Conv3d layers ──────────────────
        x = x.permute(0, 2, 1, 3, 4)   # [B, C=3, T, H=512, W=1024]

        # ── Forward through all conv layers ───────────────────────────────
        x = self.input_proj(x)          # [B,  64, T, 256, 512]
        x = self.block1(x)              # [B, 128, T, 128, 256]
        x = self.block2(x)              # [B, 256, T,  64, 128]
        x = self.block3(x)              # [B, 512, T,  32,  64]
        x = self.block4(x)              # [B, 512, T,  16,  32]
        x = self.block5(x)              # [B, 512, T,   8,  16]

        # ── Pool + flatten ─────────────────────────────────────────────────
        x = self.pool(x)                # [B, 512, 1, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 512]

        # ── Output head ────────────────────────────────────────────────────
        x = self.head(x)                # [B, 1]

        return x