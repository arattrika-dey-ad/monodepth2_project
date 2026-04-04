"""
visualise.py
------------
Helpers to render depth maps as coloured images for inspection / logging.
"""

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def colorize_depth(depth: np.ndarray,
                   cmap: str = "magma",
                   vmin: float = None,
                   vmax: float = None) -> np.ndarray:
    """
    Convert a (H, W) float depth map to a (H, W, 3) uint8 RGB image.

    Args
    ----
    depth : (H, W) depth values in metres (or disparity).
    cmap  : matplotlib colormap name (e.g. 'magma', 'plasma', 'inferno').
    vmin, vmax : colour range; defaults to 1st and 99th percentile.

    Returns
    -------
    rgb : (H, W, 3) uint8
    """
    d = depth.copy()
    if vmin is None:
        vmin = np.percentile(d, 1)
    if vmax is None:
        vmax = np.percentile(d, 99)

    d = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colormap = cm.get_cmap(cmap)
    rgb      = (colormap(d)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def tensor_to_depth_img(disp_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 1, H, W) or (1, H, W) disparity tensor to a coloured
    numpy uint8 image for TensorBoard logging.
    """
    d = disp_tensor.detach().squeeze().cpu().numpy()
    return colorize_depth(d)


def save_depth_figure(pred_depth: np.ndarray,
                      rgb_image:  np.ndarray,
                      save_path:  str,
                      gt_depth:   np.ndarray = None):
    """
    Save a side-by-side figure of RGB image + predicted depth
    (and optionally ground truth).
    """
    cols = 3 if gt_depth is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    axes[1].imshow(colorize_depth(pred_depth))
    axes[1].set_title("Predicted depth")
    axes[1].axis("off")

    if gt_depth is not None:
        axes[2].imshow(colorize_depth(gt_depth))
        axes[2].set_title("Ground truth")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
