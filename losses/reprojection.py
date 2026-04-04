"""
reprojection.py
---------------
Implements the two core loss contributions of Monodepth2:

  1. Per-pixel MINIMUM reprojection loss  (Equation 4)
     Instead of averaging over source frames, take the per-pixel minimum.
     This handles occluded pixels robustly.

  2. Auto-masking of stationary pixels    (Equation 5)
     A binary mask μ ignores pixels whose appearance does not change
     between the unwarped source image and the target — these are
     stationary objects (or a stationary camera) that provide no
     useful depth signal.

  Full-resolution multi-scale  (Section 3.2 + Fig. 3d)
     All intermediate disparity maps are UPSAMPLED to the input resolution
     BEFORE the photometric loss is computed.  This is handled in train.py.
"""

import torch
import torch.nn.functional as F
from .photometric import PhotometricLoss


# ── Module-level singleton to avoid re-instantiating ──────────────
_pe_fn = PhotometricLoss(alpha=0.85)


def compute_reprojection_loss(warped_imgs: list,
                               target_img: torch.Tensor,
                               source_imgs: list,
                               automasking: bool = True):
    """
    Compute the masked minimum-reprojection photometric loss.

    Args
    ----
    warped_imgs : list of (B, 3, H, W) — each source frame warped
                  into the target view using predicted depth + pose.
    target_img  : (B, 3, H, W) — the actual target frame I_t.
    source_imgs : list of (B, 3, H, W) — the UNWARPED source frames
                  (I_{t-1} and I_{t+1}), needed only for auto-masking.
    automasking : bool — whether to apply Equation 5.

    Returns
    -------
    loss    : scalar — mean photometric loss over unmasked pixels.
    mask    : (B, 1, H, W) — binary mask μ ∈ {0, 1}.
    min_err : (B, 1, H, W) — per-pixel minimum reprojection error
              (before masking, useful for logging).
    """
    global _pe_fn
    _pe_fn = _pe_fn.to(target_img.device)

    # ── 1. Per-pixel photometric error for each warped frame ──────────
    reprojection_losses = []
    for warped in warped_imgs:
        reprojection_losses.append(_pe_fn(warped, target_img))

    # Stack: (B, N_src, H, W)
    reprojection_losses = torch.cat(reprojection_losses, dim=1)

    # ── 2. Minimum over source frames (Equation 4) ────────────────────
    # min_err: (B, 1, H, W)
    min_err, _ = reprojection_losses.min(dim=1, keepdim=True)

    # ── 3. Auto-masking (Equation 5) ─────────────────────────────────
    if automasking and source_imgs is not None:
        identity_losses = []
        for src in source_imgs:
            identity_losses.append(_pe_fn(src, target_img))

        identity_losses = torch.cat(identity_losses, dim=1)
        # per-pixel minimum of unwarped errors
        min_identity, _ = identity_losses.min(dim=1, keepdim=True)

        # μ = 1 if warped error < unwarped error (pixel DID move → useful)
        # Add a small positive offset to break ties in favour of masking
        # identical pixels (same trick used in the official repo).
        mask = (min_err < min_identity + 1e-5).float()
    else:
        mask = torch.ones_like(min_err)

    # ── 4. Masked mean ────────────────────────────────────────────────
    masked_err = min_err * mask

    # Avoid division by zero when entire frame is masked out
    num_valid  = mask.sum() + 1e-7
    loss       = masked_err.sum() / num_valid

    return loss, mask, min_err
