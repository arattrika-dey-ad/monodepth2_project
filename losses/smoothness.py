"""
smoothness.py
-------------
Edge-aware smoothness loss (Equation 3 of the paper):

    L_s = |∂_x d*_t| exp(-|∂_x I_t|)
         + |∂_y d*_t| exp(-|∂_y I_t|)

where d*_t = d_t / mean(d_t) is the mean-normalised inverse depth.

Normalising by the mean avoids the trivial solution of shrinking
the predicted depth to zero (from DDVO, Wang et al. CVPR 2018).
"""

import torch
import torch.nn as nn


def edge_aware_smoothness(disp: torch.Tensor,
                          img: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    disp : (B, 1, H, W) — disparity (inverse depth) at ONE scale.
    img  : (B, 3, H, W) — the corresponding RGB image used to detect edges.

    Returns
    -------
    loss : scalar — mean over the batch.
    """
    # Mean-normalise disparity to prevent shrinking toward zero
    mean_disp = disp.mean(dim=[1, 2, 3], keepdim=True)
    norm_disp = disp / (mean_disp + 1e-7)           # d* = d / mean(d)

    # Spatial gradients of normalised disparity
    grad_disp_x = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
    grad_disp_y = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

    # Spatial gradients of the image (mean over channels first)
    img_mean = img.mean(dim=1, keepdim=True)          # (B, 1, H, W)
    grad_img_x = torch.abs(img_mean[:, :, :, :-1] - img_mean[:, :, :, 1:])
    grad_img_y = torch.abs(img_mean[:, :, :-1, :] - img_mean[:, :, 1:, :])

    # Edge-aware weights: exp(-|∇I|)
    smooth_x = grad_disp_x * torch.exp(-grad_img_x)
    smooth_y = grad_disp_y * torch.exp(-grad_img_y)

    return smooth_x.mean() + smooth_y.mean()
