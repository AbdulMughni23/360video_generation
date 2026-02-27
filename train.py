"""
train.py — Single entry point for Text-to-360° Video GAN training.

Run with:  python train.py

Trains a WGAN-GP with a frozen Wan2.1 backbone (Stage 1) and
trainable CNN refinement blocks (Stage 3 only).
Supports multi-GPU via nn.DataParallel across 4 x NVIDIA A10G (cuda:0-3).
Automatically resumes from the latest checkpoint in CHECKPOINT_DIR.
"""

import glob
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Project imports ────────────────────────────────────────────────────────
from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CSV_PATH,
    DATA_DIR,
    FPS,
    GPU_IDS,
    LAMBDA_BOUNDARY,
    LAMBDA_GP,
    LOG_EVERY_N_STEPS,
    LR,
    N_CRITIC,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SEED,
)
from data.dataset import VideoDataset
from model.discriminator import Discriminator
from model.generator import Generator
from model.losses import (
    gradient_penalty,
    total_generator_loss,
    wasserstein_loss_discriminator,
)
from utils.video_io import save_video

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def find_latest_checkpoint(checkpoint_dir: Path):
    """
    Scan checkpoint_dir for files named gen_epoch{N}.pt and disc_epoch{N}.pt.
    Returns the highest epoch index N if a matching pair is found, else None.
    """
    gen_files = glob.glob(str(checkpoint_dir / "gen_epoch*.pt"))
    if not gen_files:
        return None

    max_epoch = -1
    for path in gen_files:
        stem = Path(path).stem             # e.g. "gen_epoch007"
        try:
            epoch_num = int(stem.replace("gen_epoch", ""))
        except ValueError:
            continue
        disc_path = checkpoint_dir / f"disc_epoch{epoch_num:03d}.pt"
        if disc_path.exists():
            max_epoch = max(max_epoch, epoch_num)

    return max_epoch if max_epoch >= 0 else None


# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    """Main training function."""

    # ── Reproducibility ────────────────────────────────────────────────────
    set_seeds(SEED)

    # ── Run ID and output directories ──────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUTPUT_DIR / run_id).mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Run ID: %s", run_id)

    device = torch.device("cuda")

    # ── Instantiate models ─────────────────────────────────────────────────
    logger.info("Instantiating Generator …")
    generator = Generator()

    logger.info("Instantiating Discriminator …")
    discriminator = Discriminator()

    # ── Wrap in DataParallel ───────────────────────────────────────────────
    generator     = nn.DataParallel(generator,     device_ids=GPU_IDS).to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=GPU_IDS).to(device)
    logger.info("Models wrapped in DataParallel on GPUs %s", GPU_IDS)

    # ── Optimizers ─────────────────────────────────────────────────────────
    # Generator optimizer: Stage 3 CNN only
    gen_params    = [p for p in generator.module.cnn.parameters()
                     if p.requires_grad]
    gen_optimizer = torch.optim.Adam(gen_params, lr=LR, betas=(0.0, 0.9))

    # Discriminator optimizer: all parameters
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=LR, betas=(0.0, 0.9)
    )
    logger.info("Optimizers created. LR=%.2e", LR)

    # ── Mixed-precision scaler ─────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler()

    # ── DataLoader ─────────────────────────────────────────────────────────
    dataset    = VideoDataset(CSV_PATH, DATA_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    logger.info("Dataset: %d samples | Batch size: %d", len(dataset), BATCH_SIZE)

    # ── Checkpoint resume ──────────────────────────────────────────────────
    latest_epoch = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_epoch is not None:
        N = latest_epoch
        gen_ckpt  = CHECKPOINT_DIR / f"gen_epoch{N:03d}.pt"
        disc_ckpt = CHECKPOINT_DIR / f"disc_epoch{N:03d}.pt"
        generator.module.load_state_dict(torch.load(gen_ckpt, map_location=device))
        discriminator.module.load_state_dict(
            torch.load(disc_ckpt, map_location=device)
        )
        start_epoch = N + 1
        logger.info("Resumed from epoch %d", N)
    else:
        start_epoch = 0
        logger.info("Starting from scratch")

    # ── Training loop ──────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            logger.info("── Epoch %03d / %03d ──", epoch, NUM_EPOCHS - 1)

            for step, batch in enumerate(dataloader):

                real_video   = batch["real_video"].to(device)   # [B,T,C,H,W]
                descriptions = batch["description"]              # list of B str

                # ── DISCRIMINATOR UPDATES (N_CRITIC times) ─────────────────
                for _ in range(N_CRITIC):

                    # Generate fake videos — no generator gradients needed
                    with torch.no_grad():
                        fake_video = generator(descriptions)  # [B,T,C,H,W]

                    with torch.cuda.amp.autocast():
                        d_real = discriminator(real_video)    # [B,1]
                        d_fake = discriminator(fake_video)    # [B,1]
                        loss_d = wasserstein_loss_discriminator(d_real, d_fake)

                    # Gradient penalty — must be float32, outside autocast
                    with torch.cuda.amp.autocast(enabled=False):
                        gp = gradient_penalty(
                            discriminator,
                            real_video.float(),
                            fake_video.detach().float(),
                            device,
                        )

                    loss_d_total = loss_d + LAMBDA_GP * gp

                    disc_optimizer.zero_grad()
                    scaler.scale(loss_d_total).backward()
                    scaler.step(disc_optimizer)
                    scaler.update()

                # ── GENERATOR UPDATE (1 time) ──────────────────────────────
                with torch.cuda.amp.autocast():
                    fake_video = generator(descriptions)      # [B,T,C,H,W]
                    d_fake     = discriminator(fake_video)    # [B,1]

                loss_g_total, loss_g_wgan, loss_g_boundary = (
                    total_generator_loss(d_fake, fake_video, LAMBDA_BOUNDARY)
                )

                gen_optimizer.zero_grad()
                scaler.scale(loss_g_total).backward()
                scaler.step(gen_optimizer)
                scaler.update()

                # ── LOGGING ────────────────────────────────────────────────
                if step % LOG_EVERY_N_STEPS == 0:
                    logger.info(
                        "Epoch %03d | Step %05d | "
                        "D: %.4f | GP: %.4f | G: %.4f | Boundary: %.4f",
                        epoch, step,
                        loss_d_total.item(),
                        gp.item(),
                        loss_g_wgan.item(),
                        loss_g_boundary.item(),
                    )
                    print(
                        f"Epoch {epoch:03d} | Step {step:05d} | "
                        f"D: {loss_d_total.item():.4f} | "
                        f"GP: {gp.item():.4f} | "
                        f"G: {loss_g_wgan.item():.4f} | "
                        f"Boundary: {loss_g_boundary.item():.4f}"
                    )
                    videoid   = batch["videoid"][0].item()
                    save_path = (
                        OUTPUT_DIR / run_id /
                        f"{videoid}_epoch{epoch:03d}_step{step:05d}.mp4"
                    )
                    save_video(fake_video[0].detach().cpu(), save_path, fps=FPS)
                    print(f"Saved: {save_path}")

            # ── CHECKPOINT (every epoch) ───────────────────────────────────
            torch.save(
                generator.module.state_dict(),
                CHECKPOINT_DIR / f"gen_epoch{epoch:03d}.pt",
            )
            torch.save(
                discriminator.module.state_dict(),
                CHECKPOINT_DIR / f"disc_epoch{epoch:03d}.pt",
            )
            logger.info("Checkpoint saved — Epoch %03d", epoch)
            print(f"Checkpoint saved — Epoch {epoch:03d}")

    except KeyboardInterrupt:

        print("\n[STOPPED] Training interrupted.")
        print("  1 — Save full model objects  (for archiving experiment)")
        print("  2 — Save state_dicts         (for resuming training)")
        print("  3 — Save both")
        print("  4 — Exit without saving")
        choice = input("Choice (1/2/3/4): ").strip()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if choice in ("1", "3"):
            torch.save(
                generator.module,
                CHECKPOINT_DIR / f"FULL_gen_{ts}.pt",
            )
            torch.save(
                discriminator.module,
                CHECKPOINT_DIR / f"FULL_disc_{ts}.pt",
            )
            print("Full model objects saved.")

        if choice in ("2", "3"):
            torch.save(
                generator.module.state_dict(),
                CHECKPOINT_DIR / f"gen_interrupt_{ts}.pt",
            )
            torch.save(
                discriminator.module.state_dict(),
                CHECKPOINT_DIR / f"disc_interrupt_{ts}.pt",
            )
            print("State dicts saved.")

        if choice == "4":
            print("Exiting without saving.")

        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()