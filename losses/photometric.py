"""
photometric.py
--------------
Implements the photometric error function pe(I_a, I_b) from Equation (3):

    pe(I_a, I_b) = α/2 * (1 - SSIM(I_a, I_b)) + (1 - α) * |I_a - I_b|_1

where α = 0.85 (paper default).

SSIM is implemented as a 3×3 average-pooling approximation, following
the standard approach used in the official repository and Godard et al. 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM) computed on 3×3 patches
    via reflection-padded average pooling.

    Returns a per-pixel map ∈ [0, 1]; 1 = identical patches.
    """

    def __init__(self):
        super().__init__()
        self.mu_x_pool    = nn.AvgPool2d(kernel_size=3, stride=1)
        self.mu_y_pool    = nn.AvgPool2d(kernel_size=3, stride=1)
        self.sig_x_pool   = nn.AvgPool2d(kernel_size=3, stride=1)
        self.sig_y_pool   = nn.AvgPool2d(kernel_size=3, stride=1)
        self.sig_xy_pool  = nn.AvgPool2d(kernel_size=3, stride=1)

        # Stabilisation constants (standard values)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x, y : (B, 3, H, W) images with pixel values ∈ [0, 1]

        Returns
        -------
        ssim_map : (B, 3, H, W)  (smaller than input by 2 pixels each dim
                   unless padded; we pad with reflection below)
        """
        # Reflection-pad by 1 pixel to maintain spatial size
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        y = F.pad(y, (1, 1, 1, 1), mode='reflect')

        mu_x   = self.mu_x_pool(x)
        mu_y   = self.mu_y_pool(y)
        mu_x2  = mu_x ** 2
        mu_y2  = mu_y ** 2
        mu_xy  = mu_x * mu_y

        sigma_x  = self.sig_x_pool(x * x)  - mu_x2
        sigma_y  = self.sig_y_pool(y * y)  - mu_y2
        sigma_xy = self.sig_xy_pool(x * y) - mu_xy

        # SSIM formula
        num   = (2 * mu_xy    + self.C1) * (2 * sigma_xy + self.C2)
        denom = (mu_x2 + mu_y2 + self.C1) * (sigma_x + sigma_y + self.C2)

        ssim_map = torch.clamp(num / denom, min=0, max=1)
        return ssim_map


class PhotometricLoss(nn.Module):
    """
    pe(I_a, I_b) = α/2 * (1 - SSIM) + (1-α) * |L1|

    Args
    ----
    alpha : weight on the SSIM term. Paper uses 0.85.
    """

    def __init__(self, alpha: float = 0.85):
        super().__init__()
        self.alpha = alpha
        self.ssim  = SSIM()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        pred, target : (B, 3, H, W) images, pixel values ∈ [0, 1]

        Returns
        -------
        pe : (B, 1, H, W) per-pixel photometric error (mean over channels)
        """
        # SSIM term (already clamped to [0,1])
        ssim_map   = self.ssim(pred, target)                    # (B, 3, H, W)
        ssim_loss  = (1.0 - ssim_map) / 2.0                    # (B, 3, H, W)

        # L1 term
        l1_loss    = torch.abs(pred - target)                   # (B, 3, H, W)

        # Combined photometric error (mean over RGB channels)
        pe = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        pe = pe.mean(dim=1, keepdim=True)                       # (B, 1, H, W)

        return pe
