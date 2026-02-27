"""
model/losses.py — All loss functions for the Text-to-360° Video GAN.

Implements:
  - Wasserstein discriminator loss
  - Wasserstein generator loss
  - Gradient penalty (WGAN-GP)
  - Boundary consistency loss (360° seam continuity)
  - Total generator loss (combined)
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from config import BOUNDARY_PIXELS


# ──────────────────────────────────────────────────────────────────────────────
# 6.1  Wasserstein Discriminator Loss
# ──────────────────────────────────────────────────────────────────────────────

def wasserstein_loss_discriminator(d_real: Tensor, d_fake: Tensor) -> Tensor:
    """
    WGAN discriminator loss: maximise E[D(real)] - E[D(fake)].
    Equivalently, minimise E[D(fake)] - E[D(real)].

    Args:
        d_real: Discriminator scores for real videos,  shape [B, 1].
        d_fake: Discriminator scores for fake videos,  shape [B, 1].

    Returns:
        Scalar loss tensor.
    """
    return d_fake.mean() - d_real.mean()


# ──────────────────────────────────────────────────────────────────────────────
# 6.2  Wasserstein Generator Loss
# ──────────────────────────────────────────────────────────────────────────────

def wasserstein_loss_generator(d_fake: Tensor) -> Tensor:
    """
    WGAN generator loss: minimise -E[D(fake)] (fool the discriminator).

    Args:
        d_fake: Discriminator scores for generated (fake) videos, shape [B, 1].

    Returns:
        Scalar loss tensor.
    """
    return -d_fake.mean()


# ──────────────────────────────────────────────────────────────────────────────
# 6.3  Gradient Penalty (WGAN-GP)
# ──────────────────────────────────────────────────────────────────────────────

def gradient_penalty(
    discriminator: nn.Module,
    real: Tensor,
    fake: Tensor,
    device: torch.device,
) -> Tensor:
    """
    Standard WGAN-GP gradient penalty.

    Interpolates between real and fake samples, passes through the
    discriminator, and penalises deviations of the gradient norm from 1.

    IMPORTANT: Must be called OUTSIDE torch.cuda.amp.autocast() to ensure
    float32 precision for numerically stable gradient computation.

    Args:
        discriminator: The discriminator module (or DataParallel wrapper).
        real:          Real video tensor,  [B, T, C, H, W], float32.
        fake:          Fake video tensor,  [B, T, C, H, W], float32.
        device:        The device on which to create the alpha tensor.

    Returns:
        Scalar gradient penalty tensor.
    """
    B = real.size(0)

    # Sample random interpolation coefficients, one per sample in the batch
    alpha = torch.rand(B, 1, 1, 1, 1, device=device).expand_as(real)

    # Linearly interpolate between real and fake
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    # Discriminator score on interpolated samples
    d_interp = discriminator(interpolated)

    # Compute gradients of discriminator output w.r.t. interpolated inputs
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [B, T, C, H, W]

    # Flatten each sample's gradient and compute its L2 norm
    grad_norm = gradients.view(B, -1).norm(2, dim=1)  # [B]

    # Penalty: (||∇D(interpolated)||₂ - 1)²
    return ((grad_norm - 1.0) ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 6.4  Boundary Consistency Loss
# ──────────────────────────────────────────────────────────────────────────────

def boundary_consistency_loss(video: Tensor) -> Tensor:
    """
    Penalises pixel discontinuities at the 360° equirectangular seams.

    For a valid 360° video the left and right edges (horizontal seam) and the
    top and bottom edges (vertical seam) should be continuous.

    Args:
        video: FloatTensor [B, T, C, H=512, W=1024], values in [-1, 1].

    Returns:
        Scalar loss tensor.
    """
    B = BOUNDARY_PIXELS  # number of border pixels to compare (50)

    # ── Horizontal (left-right) seam ──────────────────────────────────────
    left  = video[:, :, :, :,  :B]   # leftmost  B columns
    right = video[:, :, :, :, -B:]   # rightmost B columns
    lr    = torch.mean((left - right) ** 2)

    # ── Vertical (top-bottom) seam ────────────────────────────────────────
    top    = video[:, :, :,  :B, :]  # top    B rows
    bottom = video[:, :, :, -B:, :]  # bottom B rows
    tb     = torch.mean((top - bottom) ** 2)

    return lr + tb


# ──────────────────────────────────────────────────────────────────────────────
# 6.5  Total Generator Loss
# ──────────────────────────────────────────────────────────────────────────────

def total_generator_loss(
    d_fake: Tensor,
    fake_video: Tensor,
    lambda_boundary: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Combine Wasserstein generator loss and boundary consistency loss.

    Args:
        d_fake:           Discriminator scores on generated videos, [B, 1].
        fake_video:       Generated video tensor, [B, T, C, H, W].
        lambda_boundary:  Weight for the boundary consistency term.

    Returns:
        Tuple of (total_loss, wgan_component, boundary_component).
        All are scalar tensors useful for backprop and logging.
    """
    wgan     = wasserstein_loss_generator(d_fake)
    boundary = boundary_consistency_loss(fake_video)
    total    = wgan + lambda_boundary * boundary
    return total, wgan, boundary