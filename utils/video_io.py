"""
utils/video_io.py — Video save/load utilities.

All videos are stored as .mp4 files encoded with libx264.
Tensors are float32 in [-1, 1].
"""

from pathlib import Path

import imageio
import numpy as np
import torch


def save_video(tensor: torch.Tensor, path: Path, fps: int = 20) -> None:
    """
    Save a video tensor to disk as an .mp4 file.

    Args:
        tensor: FloatTensor [T, C, H, W], float32, values in [-1, 1].
        path:   Destination .mp4 file path.
        fps:    Frames per second (default 20).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Denormalise [-1, 1] → [0, 255] uint8
    tensor = ((tensor + 1.0) * 127.5).clamp(0, 255).byte()
    # [T, C, H, W] → [T, H, W, C] for imageio
    frames = tensor.permute(0, 2, 3, 1).numpy()  # [T, H, W, C], uint8

    imageio.mimwrite(
        str(path),
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        output_params=["-pix_fmt", "yuv420p"],
    )


def load_video(path: Path) -> torch.Tensor:
    """
    Load an .mp4 video file into a tensor.

    Args:
        path: Source .mp4 file path.

    Returns:
        FloatTensor [T, C, H, W], float32, values in [-1, 1].
    """
    path = Path(path)
    reader = imageio.get_reader(str(path), "ffmpeg")

    frames = []
    for frame in reader:
        # frame is a numpy array [H, W, C], uint8
        frames.append(torch.from_numpy(np.array(frame)))
    reader.close()

    # Stack → [T, H, W, C] then permute → [T, C, H, W]
    tensor = torch.stack(frames).permute(0, 3, 1, 2).float()
    # Normalise [0, 255] → [-1, 1]
    return tensor / 127.5 - 1.0