"""
geometry.py
-----------
All camera geometry operations used during training:

  1. BackprojectDepth  — lifts pixels + depth → 3D camera-space points
  2. Project3D         — projects 3D points → normalised pixel coords [-1,1]
  3. transformation_from_parameters — axis-angle + translation → 4×4 matrix
  4. rot_from_axisangle  — axis-angle → rotation matrix (Rodrigues)

These are differentiable and run on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Axis-angle → rotation matrix (Rodrigues' formula)
# ─────────────────────────────────────────────────────────────────

def rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    """
    Convert an axis-angle representation to a 3×3 rotation matrix.
    Uses Rodrigues' formula, fully differentiable.

    Args
    ----
    vec : (B, 1, 3) axis-angle, ||vec|| = angle in radians

    Returns
    -------
    rot : (B, 4, 4) rotation matrix embedded in a 4×4 homogeneous matrix
    """
    angle = torch.norm(vec, p=2, dim=2, keepdim=True)      # (B, 1, 1)
    normalised = vec / (angle + 1e-7)                       # (B, 1, 3)

    x = normalised[..., 0].unsqueeze(2)
    y = normalised[..., 1].unsqueeze(2)
    z = normalised[..., 2].unsqueeze(2)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    t     = 1.0 - cos_a

    # Rodrigues components
    xsin = x * sin_a
    ysin = y * sin_a
    zsin = z * sin_a
    xcos = x * t
    ycos = y * t
    zcos = z * t

    batch = vec.shape[0]
    rot = torch.zeros(batch, 4, 4, device=vec.device, dtype=vec.dtype)

    rot[:, 0, 0] = (t * x * x + cos_a).view(-1)
    rot[:, 0, 1] = (xcos * y - zsin).view(-1)
    rot[:, 0, 2] = (xcos * z + ysin).view(-1)
    rot[:, 1, 0] = (ycos * x + zsin).view(-1)
    rot[:, 1, 1] = (t * y * y + cos_a).view(-1)
    rot[:, 1, 2] = (ycos * z - xsin).view(-1)
    rot[:, 2, 0] = (zcos * x - ysin).view(-1)
    rot[:, 2, 1] = (zcos * y + xsin).view(-1)
    rot[:, 2, 2] = (t * z * z + cos_a).view(-1)
    rot[:, 3, 3] = 1.0

    return rot


def transformation_from_parameters(axisangle: torch.Tensor,
                                   translation: torch.Tensor,
                                   invert: bool = False) -> torch.Tensor:
    """
    Convert pose decoder output → 4×4 transformation matrix.

    Args
    ----
    axisangle   : (B, 1, 3)
    translation : (B, 1, 3)
    invert      : if True, invert the pose (t' → t instead of t → t')

    Returns
    -------
    T : (B, 4, 4)
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = R.clone()
    T[:, :3, 3] = t.squeeze(1)    # embed translation in last column

    return T


# ─────────────────────────────────────────────────────────────────
# BackprojectDepth
# ─────────────────────────────────────────────────────────────────

class BackprojectDepth(nn.Module):
    """
    Lifts every pixel in a depth map to a 3D point in camera space.

    Given:
        D(u,v)   = depth at pixel (u,v)
        K^{-1}   = inverse camera intrinsics

    Computes:
        P_cam = D * K^{-1} * [u, v, 1]^T

    This creates the pixel-grid once in __init__ for efficiency.
    """

    def __init__(self, batch_size: int, height: int, width: int):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width  = width

        # Build a homogeneous pixel grid [u, v, 1] of shape (3, H*W)
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)   # (2, H, W)
        id_coords = torch.from_numpy(id_coords)

        ones = torch.ones(1, height, width, dtype=torch.float32)
        pix_coords = torch.cat([id_coords, ones], dim=0)             # (3, H, W)
        pix_coords = pix_coords.view(1, 3, -1)                       # (1, 3, H*W)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)             # (B, 3, H*W)

        self.register_buffer('pix_coords', pix_coords)

    def forward(self, depth: torch.Tensor,
                inv_K: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        depth : (B, 1, H, W)
        inv_K : (B, 4, 4) inverse intrinsics

        Returns
        -------
        cam_points : (B, 4, H*W) homogeneous 3D points
        """
        cam_points = torch.matmul(
            inv_K[:, :3, :3],          # (B, 3, 3)
            self.pix_coords            # (B, 3, H*W)
        )                              # (B, 3, H*W)

        depth_flat = depth.view(self.batch_size, 1, -1)  # (B, 1, H*W)
        cam_points = depth_flat * cam_points             # (B, 3, H*W)

        # Append a row of 1s → homogeneous (B, 4, H*W)
        ones = torch.ones(
            self.batch_size, 1, self.height * self.width,
            device=depth.device, dtype=depth.dtype
        )
        cam_points = torch.cat([cam_points, ones], dim=1)

        return cam_points


# ─────────────────────────────────────────────────────────────────
# Project3D
# ─────────────────────────────────────────────────────────────────

class Project3D(nn.Module):
    """
    Projects 3D camera-space points into a source image using
    the relative camera transformation T and intrinsics K.

    Returns normalised coordinates in [-1, 1] suitable for
    F.grid_sample().
    """

    def __init__(self, batch_size: int, height: int, width: int,
                 eps: float = 1e-7):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width  = width
        self.eps    = eps

    def forward(self,
                points: torch.Tensor,
                K: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        points : (B, 4, H*W) homogeneous 3D points in target-cam space
        K      : (B, 4, 4) camera intrinsics
        T      : (B, 4, 4) transformation T_{t → t'} (target → source)

        Returns
        -------
        pix_coords : (B, H, W, 2) normalised pixel coords ∈ [-1, 1]
        """
        # Transform points into source-camera space
        P = torch.matmul(K, T)[:, :3, :]   # (B, 3, 4)

        # Project
        cam_points = torch.matmul(P, points)   # (B, 3, H*W)

        # Perspective division
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2:3, :] + self.eps)

        # Reshape to (B, 2, H, W)
        pix_coords = pix_coords.view(
            self.batch_size, 2, self.height, self.width
        )

        # Normalise to [-1, 1] for grid_sample
        pix_coords = pix_coords.permute(0, 2, 3, 1)   # (B, H, W, 2)
        pix_coords[..., 0] /= (self.width  - 1)
        pix_coords[..., 1] /= (self.height - 1)
        pix_coords = (pix_coords - 0.5) * 2.0

        return pix_coords
