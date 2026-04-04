"""
transforms.py
-------------
All data augmentation transforms used during training of Monodepth2.

Paper details (Section 3.3):
  • Horizontal flips with 50% probability.
  • Colour augmentation (brightness, contrast, saturation, hue jitter)
    with 50% probability — BUT only applied to the images fed to the
    networks ("color_aug"), NOT to those used to compute the photometric
    loss ("color").
  • All three frames in a triplet (I_{t-1}, I_t, I_{t+1}) receive the
    SAME colour augmentation parameters so that the relative appearance
    between frames is consistent.

Augmentation ranges from the paper:
  brightness : ±0.2
  contrast   : ±0.2
  saturation : ±0.2
  hue        : ±0.1

These are implemented as composable callable classes so they can be
applied selectively (e.g. colour jitter only to the network input).
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ═════════════════════════════════════════════════════════════════
# Individual transform building blocks
# ═════════════════════════════════════════════════════════════════

class ResizeImage:
    """
    Resize a PIL image to (height, width) using Lanczos resampling.

    Lanczos is the default used by the official Monodepth2 repo and
    produces better quality than bilinear for downsampling.

    Args
    ----
    height : int – target height in pixels.
    width  : int – target width in pixels.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width  = width
        self.resize = T.Resize(
            (height, width),
            interpolation=T.InterpolationMode.LANCZOS
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.resize(img)

    def __repr__(self) -> str:
        return f"ResizeImage(h={self.height}, w={self.width})"


# ─────────────────────────────────────────────────────────────────

class RandomHorizontalFlip:
    """
    Randomly flip a PIL image (or tensor) horizontally with probability p.

    Tracks whether a flip was applied so the same decision can be
    propagated to all frames in a triplet and to the intrinsics matrix.

    Usage
    -----
        flipper = RandomHorizontalFlip(p=0.5)
        do_flip = flipper.decide()            # sample once per triplet
        flipped = flipper.apply(img, do_flip) # apply to each frame
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def decide(self) -> bool:
        """Sample the flip decision for one data point."""
        return random.random() < self.p

    def apply(self, img, do_flip: bool):
        """
        Apply (or skip) the flip.

        Args
        ----
        img     : PIL Image or (C, H, W) torch.Tensor.
        do_flip : bool — result of decide().

        Returns
        -------
        Same type as input.
        """
        if not do_flip:
            return img
        if isinstance(img, Image.Image):
            return TF.hflip(img)
        elif isinstance(img, torch.Tensor):
            return torch.flip(img, dims=[2])   # flip W dimension
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def __call__(self, img) -> tuple:
        """
        Convenience: decide AND apply in one call.

        Returns
        -------
        (flipped_img, did_flip)
        """
        do_flip = self.decide()
        return self.apply(img, do_flip), do_flip

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


# ─────────────────────────────────────────────────────────────────

class ColorJitterDeterministic:
    """
    Apply colour jitter with PRE-SAMPLED parameters.

    This is the key difference from a standard random colour jitter:
    all frames in a triplet must receive the SAME colour transform
    so that the photometric error between frames reflects true 3D
    geometry rather than artificially introduced colour differences.

    Usage
    -----
        jitter = ColorJitterDeterministic(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        # Sample once per triplet:
        params = jitter.sample_params()
        # Apply to every frame:
        aug_frame = jitter.apply(frame, params)

    Args
    ----
    brightness : float – max brightness delta (uniform in [-b, +b]).
    contrast   : float – max contrast delta.
    saturation : float – max saturation delta.
    hue        : float – max hue delta (uniform in [-h, +h]; must ≤ 0.5).
    p          : float – probability of applying any colour jitter at all.
    """

    def __init__(self,
                 brightness: float = 0.2,
                 contrast:   float = 0.2,
                 saturation: float = 0.2,
                 hue:        float = 0.1,
                 p:          float = 0.5):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue
        self.p          = p

        # Build the underlying torchvision jitter to borrow its
        # parameter sampler.
        self._jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def sample_params(self) -> Optional[dict]:
        """
        Sample jitter parameters for one triplet.

        Returns
        -------
        dict with keys: 'apply', 'brightness', 'contrast',
                        'saturation', 'hue', 'order'
        OR None (with probability 1-p) meaning no jitter.
        """
        if random.random() > self.p:
            return {"apply": False}

        # Sample each factor uniformly in [1-delta, 1+delta]
        def sample_factor(delta: float) -> float:
            return random.uniform(max(0.0, 1.0 - delta), 1.0 + delta)

        def sample_hue(delta: float) -> float:
            return random.uniform(-delta, delta)

        # Random order of transform application (matches torchvision)
        order = list(range(4))
        random.shuffle(order)

        return {
            "apply":      True,
            "brightness": sample_factor(self.brightness),
            "contrast":   sample_factor(self.contrast),
            "saturation": sample_factor(self.saturation),
            "hue":        sample_hue(self.hue),
            "order":      order,
        }

    def apply(self, img: Image.Image,
              params: Optional[dict]) -> Image.Image:
        """
        Apply pre-sampled jitter to a PIL image.

        Args
        ----
        img    : PIL Image (RGB).
        params : dict returned by sample_params(), or None.

        Returns
        -------
        PIL Image (augmented or original).
        """
        if params is None or not params.get("apply", False):
            return img

        transforms_list = [
            lambda x: TF.adjust_brightness(x, params["brightness"]),
            lambda x: TF.adjust_contrast(x,   params["contrast"]),
            lambda x: TF.adjust_saturation(x, params["saturation"]),
            lambda x: TF.adjust_hue(x,        params["hue"]),
        ]

        for idx in params["order"]:
            img = transforms_list[idx](img)

        return img

    def __repr__(self) -> str:
        return (f"ColorJitterDeterministic("
                f"brightness={self.brightness}, contrast={self.contrast}, "
                f"saturation={self.saturation}, hue={self.hue}, p={self.p})")


# ─────────────────────────────────────────────────────────────────

class ToTensor:
    """
    Convert a PIL Image to a (C, H, W) float32 tensor with values in [0, 1].
    Thin wrapper around torchvision.transforms.ToTensor.
    """

    def __init__(self):
        self._to_tensor = T.ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(img)

    def __repr__(self) -> str:
        return "ToTensor()"


# ═════════════════════════════════════════════════════════════════
# Intrinsics helpers
# ═════════════════════════════════════════════════════════════════

def adjust_intrinsics_for_flip(K: torch.Tensor,
                                width: int) -> torch.Tensor:
    """
    When an image is horizontally flipped, the principal point cx
    moves from cx to (W - 1 - cx).

    Args
    ----
    K     : (4, 4) camera intrinsics matrix.
    width : int – image width W.

    Returns
    -------
    K_flipped : (4, 4) updated intrinsics.
    """
    K = K.clone()
    K[0, 2] = width - 1 - K[0, 2]
    return K


def adjust_intrinsics_for_crop(K: torch.Tensor,
                                crop_top: int,
                                crop_left: int) -> torch.Tensor:
    """
    When an image is cropped (top-left corner at (crop_left, crop_top)),
    the principal point shifts accordingly.

    Args
    ----
    K          : (4, 4) camera intrinsics matrix.
    crop_top   : int – number of pixels removed from the top.
    crop_left  : int – number of pixels removed from the left.

    Returns
    -------
    K_cropped : (4, 4) updated intrinsics.
    """
    K = K.clone()
    K[0, 2] -= crop_left
    K[1, 2] -= crop_top
    return K


def adjust_intrinsics_for_scale(K: torch.Tensor,
                                 orig_height: int,
                                 orig_width:  int,
                                 new_height:  int,
                                 new_width:   int) -> torch.Tensor:
    """
    Scale the intrinsics when the image is resized from
    (orig_height, orig_width) → (new_height, new_width).

    fx_new = fx_orig * (new_width  / orig_width)
    fy_new = fy_orig * (new_height / orig_height)
    cx_new = cx_orig * (new_width  / orig_width)
    cy_new = cy_orig * (new_height / orig_height)

    Args
    ----
    K                        : (4, 4) original intrinsics.
    orig_height, orig_width  : original image resolution.
    new_height,  new_width   : target   image resolution.

    Returns
    -------
    K_scaled : (4, 4) updated intrinsics.
    """
    K = K.clone()
    sx = new_width  / orig_width
    sy = new_height / orig_height
    K[0, 0] *= sx   # fx
    K[1, 1] *= sy   # fy
    K[0, 2] *= sx   # cx
    K[1, 2] *= sy   # cy
    return K


# ═════════════════════════════════════════════════════════════════
# Composed training / validation transforms
# ═════════════════════════════════════════════════════════════════

class TrainTransforms:
    """
    Composed transform pipeline for one training triplet.

    Applies, in order:
      1. Resize all frames to (height, width).
      2. Optionally flip all frames horizontally (same decision per triplet).
      3. Convert all frames to tensors.
      4. Optionally apply identical colour jitter to the "aug" copies.

    The output is a dict update that can be merged into the inputs dict
    produced by KITTIDataset.

    Usage
    -----
        tf = TrainTransforms(height=192, width=640)
        pil_frames  = {-1: img_prev, 0: img_curr, 1: img_next}
        K           = build_intrinsics(192, 640)
        result      = tf(pil_frames, K)
        # result contains:
        #   ("color",     fi, 0) → tensor (no colour aug)
        #   ("color_aug", fi, 0) → tensor (with colour aug)
        #   "K"                  → (4,4) tensor
        #   "inv_K"              → (4,4) tensor
    """

    def __init__(self,
                 height:     int   = 192,
                 width:      int   = 640,
                 flip_p:     float = 0.5,
                 jitter_p:   float = 0.5,
                 brightness: float = 0.2,
                 contrast:   float = 0.2,
                 saturation: float = 0.2,
                 hue:        float = 0.1):

        self.height = height
        self.width  = width

        self.resizer  = ResizeImage(height, width)
        self.flipper  = RandomHorizontalFlip(p=flip_p)
        self.jitter   = ColorJitterDeterministic(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=jitter_p,
        )
        self.to_tensor = ToTensor()

    def __call__(self,
                 pil_frames: dict,
                 K: torch.Tensor) -> dict:
        """
        Args
        ----
        pil_frames : dict mapping frame_offset (int) → PIL Image.
                     Keys should include at minimum 0, -1, 1.
        K          : (4, 4) camera intrinsics for the ORIGINAL image size.

        Returns
        -------
        result : dict containing tensors for all frames + intrinsics.
        """
        result = {}

        # ── 1. Decide augmentation parameters (once per triplet) ──────
        do_flip    = self.flipper.decide()
        jitter_params = self.jitter.sample_params()

        # ── 2. Process each frame ─────────────────────────────────────
        for fi, pil_img in pil_frames.items():
            # Resize
            img = self.resizer(pil_img)

            # Horizontal flip
            img = self.flipper.apply(img, do_flip)

            # Colour-augmented copy (fed to networks)
            img_aug = self.jitter.apply(img, jitter_params)

            # Convert to tensor
            result[("color",     fi, 0)] = self.to_tensor(img)
            result[("color_aug", fi, 0)] = self.to_tensor(img_aug)

        # ── 3. Adjust intrinsics ──────────────────────────────────────
        K_adj = K.clone()
        if do_flip:
            K_adj = adjust_intrinsics_for_flip(K_adj, self.width)

        result["K"]     = K_adj
        result["inv_K"] = torch.inverse(K_adj)

        return result


class ValTransforms:
    """
    Deterministic transform pipeline for validation / test.

    Only resizes and converts to tensor — no flipping or colour jitter.
    The "color" and "color_aug" copies are identical.
    """

    def __init__(self, height: int = 192, width: int = 640):
        self.height    = height
        self.width     = width
        self.resizer   = ResizeImage(height, width)
        self.to_tensor = ToTensor()

    def __call__(self,
                 pil_frames: dict,
                 K: torch.Tensor) -> dict:
        result = {}

        for fi, pil_img in pil_frames.items():
            img = self.resizer(pil_img)
            t   = self.to_tensor(img)
            result[("color",     fi, 0)] = t
            result[("color_aug", fi, 0)] = t.clone()

        result["K"]     = K.clone()
        result["inv_K"] = torch.inverse(K)

        return result
