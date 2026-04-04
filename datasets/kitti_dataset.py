"""
kitti_dataset.py
----------------
PyTorch Dataset class for self-supervised monocular depth training.

Loads TEMPORAL TRIPLETS:  (I_{t-1}, I_t, I_{t+1})

Paper details (Section 3.3 + Phase 2 of the assignment):
  • Input resolution: 640 × 192 (width × height).
  • Colour augmentation (brightness ±0.2, contrast ±0.2,
    saturation ±0.2, hue ±0.1) applied with 50% probability,
    BUT ONLY to the images fed to the networks,
    NOT to those used to compute the photometric loss.
  • Horizontal flips with 50% probability.

Directory structure expected:

    <data_path>/
        <sequence_folder>/
            image_02/data/   ← left colour camera frames
        splits/
            toy_train.txt    ← lines: "seq_folder frame_idx"
            toy_val.txt
            toy_test.txt

Intrinsics: uses a simplified camera model with a single set of
intrinsics for all KITTI images (as in the paper, Section 4.1):
  principal point = image centre
  focal length    = average KITTI focal length ≈ 0.58 × width
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────
# Approximate KITTI intrinsics used in the paper
# (These match the avg focal length used by Zhou et al.)
# ─────────────────────────────────────────────────────────────────
KITTI_AVG_FOCAL = 0.58   # fx = fy = KITTI_AVG_FOCAL * width


def build_intrinsics(height: int, width: int) -> np.ndarray:
    """
    Returns a 4×4 camera intrinsics matrix K.
    Uses the averaged focal length from the paper.
    """
    fx = KITTI_AVG_FOCAL * width
    fy = KITTI_AVG_FOCAL * width
    cx = 0.5 * width
    cy = 0.5 * height

    K = np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0, 1],
    ], dtype=np.float32)
    return K


class KITTIDataset(Dataset):
    """
    Args
    ----
    data_path  : root of the toy/kitti dataset.
    split_file : path to a txt file with lines "seq_folder frame_idx".
    height, width : target image dimensions.
    is_train   : if True, apply data augmentation.
    frame_idxs : which relative frame offsets to load, e.g. [0, -1, 1].
    """

    def __init__(self,
                 data_path: str,
                 split_file: str,
                 height: int = 192,
                 width:  int = 640,
                 is_train: bool = True,
                 frame_idxs: list = [0, -1, 1]):
        super().__init__()

        self.data_path  = Path(data_path)
        self.height     = height
        self.width      = width
        self.is_train   = is_train
        self.frame_idxs = frame_idxs

        # ── Load split ────────────────────────────────────────────────
        with open(split_file) as f:
            lines = f.read().splitlines()

        # Each line: "sequence_folder frame_idx"
        self.filenames = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                seq, idx = parts[0], int(parts[1])
                self.filenames.append((seq, idx))

        # ── Intrinsics ────────────────────────────────────────────────
        K    = build_intrinsics(height, width)
        invK = np.linalg.inv(K)

        self.K    = torch.from_numpy(K)
        self.invK = torch.from_numpy(invK)

        # ── Image transforms ──────────────────────────────────────────
        self.to_tensor = transforms.ToTensor()
        self.resize    = transforms.Resize((height, width),
                                           interpolation=transforms.InterpolationMode.LANCZOS)

        # Colour jitter parameters from the paper
        self.brightness = 0.2
        self.contrast   = 0.2
        self.saturation = 0.2
        self.hue        = 0.1

    # ──────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.filenames)

    # ──────────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        seq, frame_idx = self.filenames[idx]

        # ── Decide augmentation parameters for this sample ────────────
        do_flip   = self.is_train and random.random() > 0.5
        do_colour = self.is_train and random.random() > 0.5

        if do_colour:
            colour_jitter = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
            colour_aug_fn = transforms.Compose([colour_jitter])
        else:
            colour_aug_fn = None

        # ── Load images ───────────────────────────────────────────────
        inputs = {}
        for fi in self.frame_idxs:
            img = self._load_image(seq, frame_idx + fi)
            if img is None:
                # If boundary frame missing, repeat the centre frame
                img = self._load_image(seq, frame_idx)

            img = self.resize(img)

            # "colour_aug" = fed to networks (with colour jitter)
            # "colour"     = used to compute photometric loss (NO jitter)
            img_tensor = self.to_tensor(img)

            if do_flip:
                img_tensor = torch.flip(img_tensor, dims=[2])

            inputs[("color", fi, 0)] = img_tensor   # unaugmented

            if colour_aug_fn is not None:
                img_pil  = transforms.ToPILImage()(img_tensor)
                img_aug  = colour_aug_fn(img_pil)
                img_aug  = self.to_tensor(img_aug)
            else:
                img_aug = img_tensor.clone()

            inputs[("color_aug", fi, 0)] = img_aug  # augmented → fed to networks

        # ── Intrinsics ────────────────────────────────────────────────
        K    = self.K.clone()
        invK = self.invK.clone()

        if do_flip:
            # Flip the principal point cx
            K[0, 2] = self.width - 1 - K[0, 2]
            invK = torch.inverse(K)

        inputs["K"]    = K
        inputs["inv_K"] = invK

        return inputs

    # ──────────────────────────────────────────────────────────────────
    def _load_image(self, seq: str, frame_idx: int):
        """
        Load one frame from the left colour camera (image_02).
        Returns PIL Image or None if the file does not exist.
        """
        img_path = (
            self.data_path / seq / "image_02" / "data"
            / f"{frame_idx:010d}.png"
        )
        if not img_path.exists():
            # Try .jpg fallback
            img_path = img_path.with_suffix(".jpg")
        if not img_path.exists():
            return None
        return Image.open(img_path).convert("RGB")
