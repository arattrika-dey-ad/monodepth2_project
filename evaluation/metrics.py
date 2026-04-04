"""
metrics.py
----------
Standard depth estimation evaluation metrics used in the paper (Table 1):

  Error metrics (lower is better):
    • Abs Rel   = mean |d - d*| / d*
    • Sq Rel    = mean (d - d*)² / d*
    • RMSE      = sqrt(mean (d - d*)²)
    • RMSE log  = sqrt(mean (log d - log d*)²)

  Accuracy metrics (higher is better):
    • δ < 1.25       — fraction of pixels where max(d/d*, d*/d) < 1.25
    • δ < 1.25²
    • δ < 1.25³

All metrics are computed AFTER applying median scaling (for monocular),
and after capping depth at 80 m (standard KITTI evaluation practice).

Median scaling (Zhou et al. 2017):
    scale = median(gt_depth) / median(pred_depth)
    pred  = pred * scale
"""

import numpy as np


def compute_depth_errors(gt: np.ndarray,
                         pred: np.ndarray) -> dict:
    """
    Compute all standard depth evaluation metrics.

    Args
    ----
    gt   : (N,) ground truth depths (valid pixels only, in metres).
    pred : (N,) predicted depths (same length, already scaled).

    Returns
    -------
    dict of metric_name → float
    """
    thresh = np.maximum(gt / pred, pred / gt)   # element-wise max ratio

    a1 = (thresh < 1.25    ).mean()
    a2 = (thresh < 1.25**2 ).mean()
    a3 = (thresh < 1.25**3 ).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel  = np.mean((gt - pred) ** 2  / gt)
    rmse    = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(
        np.mean((np.log(gt) - np.log(pred)) ** 2)
    )

    return {
        "abs_rel":  abs_rel,
        "sq_rel":   sq_rel,
        "rmse":     rmse,
        "rmse_log": rmse_log,
        "a1":       a1,
        "a2":       a2,
        "a3":       a3,
    }


def apply_median_scaling(pred: np.ndarray,
                         gt: np.ndarray) -> np.ndarray:
    """
    Apply per-image median scaling.

    scale = median(gt_valid) / median(pred_valid)
    pred_scaled = pred * scale

    Args
    ----
    pred : (H, W) predicted depth map.
    gt   : (H, W) ground truth depth map (0 = invalid).

    Returns
    -------
    pred_scaled : (H, W)
    """
    valid_mask = gt > 0
    if valid_mask.sum() == 0:
        return pred

    scale = np.median(gt[valid_mask]) / (np.median(pred[valid_mask]) + 1e-7)
    return pred * scale


def evaluate_batch(preds: list,
                   gts: list,
                   min_depth: float = 1e-3,
                   max_depth: float = 80.0,
                   use_median_scaling: bool = True) -> dict:
    """
    Evaluate a list of predicted depth maps against ground-truth.

    Args
    ----
    preds : list of (H, W) numpy arrays — predicted depths.
    gts   : list of (H, W) numpy arrays — ground-truth depths.
    min_depth, max_depth : clipping range (standard KITTI: 1e-3 to 80 m).
    use_median_scaling   : apply per-image median scaling (required for M).

    Returns
    -------
    mean metrics dict, and per-image list of dicts.
    """
    all_metrics = []

    for pred, gt in zip(preds, gts):
        # Resize prediction to ground truth resolution if needed
        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        # Median scaling
        if use_median_scaling:
            pred = apply_median_scaling(pred, gt)

        # Mask invalid GT pixels and clip depth range
        mask = (gt > min_depth) & (gt < max_depth)
        if mask.sum() == 0:
            continue

        gt_valid   = gt[mask]
        pred_valid = np.clip(pred[mask], min_depth, max_depth)

        metrics = compute_depth_errors(gt_valid, pred_valid)
        all_metrics.append(metrics)

    if not all_metrics:
        return {}, []

    # Average across images
    mean_metrics = {}
    for key in all_metrics[0]:
        mean_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    return mean_metrics, all_metrics


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print a metrics dict."""
    header = f"{prefix}  " if prefix else ""
    print(
        f"{header}"
        f"AbsRel={metrics['abs_rel']:.4f}  "
        f"SqRel={metrics['sq_rel']:.4f}  "
        f"RMSE={metrics['rmse']:.3f}  "
        f"RMSElog={metrics['rmse_log']:.4f}  "
        f"δ<1.25={metrics['a1']:.4f}  "
        f"δ<1.25²={metrics['a2']:.4f}  "
        f"δ<1.25³={metrics['a3']:.4f}"
    )
