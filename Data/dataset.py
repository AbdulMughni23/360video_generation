"""
data/dataset.py — Dataset and DataLoader for 360° video GAN training.

Reads a CSV with columns 'videoid' and 'description', loads the corresponding
.mp4 files, and returns normalized video tensors plus their text descriptions.
"""

import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def _load_video_decord(video_path: Path) -> torch.Tensor:
    """
    Load video using decord.
    Returns: [T=100, C=3, H=512, W=1024], float32, normalised to [-1, 1].
    """
    import decord
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    frames = vr.get_batch(list(range(100))).asnumpy()   # [T, H, W, C]
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [T,C,H,W]
    tensor = tensor / 127.5 - 1.0
    return tensor


def _load_video_torchvision(video_path: Path) -> torch.Tensor:
    """
    Fallback loader using torchvision.io.read_video.
    Returns: [T=100, C=3, H=512, W=1024], float32, normalised to [-1, 1].
    """
    import torchvision.io as tvio
    # read_video returns (video, audio, info); video is [T, H, W, C] uint8
    video, _, _ = tvio.read_video(str(video_path), output_format="TCHW")
    # video is already [T, C, H, W] with output_format="TCHW"
    video = video[:100].float()                          # take first 100 frames
    video = video / 127.5 - 1.0
    return video


def load_video(video_path: Path) -> torch.Tensor:
    """
    Load a video file, preferring decord and falling back to torchvision.
    Returns: [T=100, C=3, H=512, W=1024], float32, [-1, 1].
    """
    try:
        return _load_video_decord(video_path)
    except ImportError:
        logger.warning("decord not available — falling back to torchvision")
        return _load_video_torchvision(video_path)


class VideoDataset(Dataset):
    """
    Dataset of real 360° videos paired with text descriptions.

    CSV must have columns:
        - 'videoid'     (int)  — numeric video ID
        - 'description' (str)  — text prompt

    Video filename: DATA_DIR / f"{videoid}.mp4"
    Missing video files are skipped with a logged warning.
    """

    def __init__(self, csv_path: Union[str, Path], data_dir: Union[str, Path]):
        """
        Args:
            csv_path: Path to the metadata CSV file.
            data_dir: Directory containing the .mp4 video files.
        """
        self.data_dir = Path(data_dir)
        df = pd.read_csv(str(csv_path))

        # Validate required columns
        required_cols = {"videoid", "description"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        # Filter rows whose video file actually exists
        self.samples = []
        for _, row in df.iterrows():
            videoid = int(row["videoid"])
            description = str(row["description"])
            video_path = self.data_dir / f"{videoid}.mp4"
            if not video_path.exists():
                logger.warning(
                    f"Video file not found, skipping: {video_path}"
                )
                continue
            self.samples.append({"videoid": videoid, "description": description,
                                  "video_path": video_path})

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid video samples found. "
                "Check DATA_DIR and CSV_PATH in config.py."
            )

        logger.info(
            f"VideoDataset: {len(self.samples)} valid samples loaded "
            f"from {csv_path}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dict with:
            'real_video':  FloatTensor [T=100, C=3, H=512, W=1024], [-1,1]
            'description': str
            'videoid':     int
        """
        sample = self.samples[idx]
        video_path = sample["video_path"]

        # Load and normalise video
        real_video = load_video(video_path)   # [T=100, C=3, H=512, W=1024]

        return {
            "real_video":  real_video,
            "description": sample["description"],
            "videoid":     sample["videoid"],
        }


def build_dataloader(csv_path, data_dir, batch_size: int) -> DataLoader:
    """
    Convenience function: create DataLoader from config values.
    """
    dataset = VideoDataset(csv_path=csv_path, data_dir=data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )