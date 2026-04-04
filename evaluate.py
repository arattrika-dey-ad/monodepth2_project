"""
evaluate.py
-----------
Evaluate a trained Monodepth2 (from-scratch) checkpoint on a test split.

Reports the 7 standard metrics from the paper (Table 1):
  Abs Rel, Sq Rel, RMSE, RMSE log, δ<1.25, δ<1.25², δ<1.25³

Applies PER-IMAGE MEDIAN SCALING (required for monocular training):
    scale = median(gt_valid) / median(pred_valid)
    pred  = pred * scale

Ground-truth depth is read from KITTI velodyne projected depth maps
(PNG files with uint16 encoding, millimetres / 256).

Usage
-----
python evaluate.py \
    --checkpoint  checkpoints/scratch_toy/epoch_19.pth \
    --data_path   /data/kitti_toy \
    --split_file  /data/kitti_toy/splits/toy_test.txt \
    --height      192 \
    --width       640

For comparison with the official model:
python evaluate.py \
    --checkpoint  <path_to_official_weights> \
    --data_path   /data/kitti_toy \
    --split_file  /data/kitti_toy/splits/toy_test.txt \
    --official    # uses the official model loading code
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models   import ResNetEncoder, DepthDecoder
from datasets import KITTIDataset
from evaluation import evaluate_batch, print_metrics


# ═════════════════════════════════════════════════════════════════
# Depth ground-truth loader
# ═════════════════════════════════════════════════════════════════

def load_gt_depth(seq: str, frame_idx: int,
                  data_path: Path) -> np.ndarray:
    """
    Load a KITTI velodyne projected depth map.

    KITTI stores projected velodyne depth as uint16 PNG images
    where depth_metres = pixel_value / 256.0.

    Returns
    -------
    depth : (H, W) float32 array; 0 = invalid.
    """
    import cv2
    depth_path = (
        data_path / seq / "proj_depth" / "groundtruth"
        / "image_02" / f"{frame_idx:010d}.png"
    )
    if not depth_path.exists():
        # Fall back: no GT available
        return None

    depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        return None

    depth = depth_png.astype(np.float32) / 256.0
    return depth


# ═════════════════════════════════════════════════════════════════
# Argument parsing
# ═════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Monodepth2 evaluation")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data_path",   type=str, required=True)
    p.add_argument("--split_file",  type=str, required=True)
    p.add_argument("--height",      type=int, default=192)
    p.add_argument("--width",       type=int, default=640)
    p.add_argument("--min_depth",   type=float, default=1e-3)
    p.add_argument("--max_depth",   type=float, default=80.0)
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_median_scaling", action="store_true")
    p.add_argument("--output_json", type=str, default=None,
                   help="If set, write metrics to this JSON file.")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════
# Load model from checkpoint
# ═════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, height: int, width: int, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Infer pretrained from checkpoint keys (fallback: True)
    depth_encoder = ResNetEncoder(
        num_layers=18, pretrained=False, num_input_images=1
    ).to(device)
    depth_decoder = DepthDecoder(
        num_ch_enc=depth_encoder.num_ch_enc, scales=[0]
    ).to(device)

    depth_encoder.load_state_dict(ckpt["depth_encoder"])
    depth_decoder.load_state_dict(ckpt["depth_decoder"])

    depth_encoder.eval()
    depth_decoder.eval()
    return depth_encoder, depth_decoder


# ═════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═════════════════════════════════════════════════════════════════

def main():
    opt    = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Evaluate] Loading checkpoint: {opt.checkpoint}")
    depth_encoder, depth_decoder = load_model(
        opt.checkpoint, opt.height, opt.width, device
    )

    # Dataset (no augmentation, only centre frame needed for depth)
    dataset = KITTIDataset(
        data_path=opt.data_path,
        split_file=opt.split_file,
        height=opt.height,
        width=opt.width,
        is_train=False,
        frame_idxs=[0],
    )
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    all_preds = []
    all_gts   = []

    data_path = Path(opt.data_path)

    # Parse split file to get seq + frame_idx for GT lookup
    with open(opt.split_file) as f:
        split_lines = f.read().splitlines()
    split_entries = []
    for line in split_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            split_entries.append((parts[0], int(parts[1])))

    print(f"[Evaluate] Running on {len(dataset)} frames...")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(loader)):
            img = inputs[("color", 0, 0)].to(device)

            # Forward pass: get finest-scale (scale=0) depth
            feats     = depth_encoder(img)
            depth_out = depth_decoder(feats)
            disp      = depth_out[("disp", 0)]   # (B, 1, H, W)

            # Convert disparity → depth
            min_d, max_d = 0.1, 100.0
            a = 1.0 / min_d - 1.0 / max_d
            b = 1.0 / max_d
            pred_depth = (1.0 / (a * disp + b)).squeeze(1)   # (B, H, W)

            # Accumulate predictions and GT
            bs = img.shape[0]
            for i in range(bs):
                global_idx = batch_idx * opt.batch_size + i
                if global_idx >= len(split_entries):
                    break

                seq, frame_idx = split_entries[global_idx]
                pred_np = pred_depth[i].cpu().numpy()   # (H, W)

                # Load GT (may be None for toy data without GT)
                gt_np = load_gt_depth(seq, frame_idx, data_path)
                if gt_np is None:
                    # Synthetic GT fallback for demo purposes
                    # In a real run this would be skipped
                    continue

                # Resize GT to prediction resolution if needed
                if gt_np.shape != pred_np.shape:
                    import cv2
                    gt_np = cv2.resize(
                        gt_np,
                        (pred_np.shape[1], pred_np.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                all_preds.append(pred_np)
                all_gts.append(gt_np)

    if not all_preds:
        print("\n[Evaluate] No ground-truth depth found.")
        print("  → If you are using a toy dataset without GT depth maps,")
        print("     use the qualitative visualisation instead (infer.py).")
        print("  → See README for how to obtain GT depth from KITTI.")
        return

    print(f"\n[Evaluate] Computing metrics on {len(all_preds)} frames...")

    use_scaling = not opt.no_median_scaling
    mean_metrics, per_img = evaluate_batch(
        preds=all_preds,
        gts=all_gts,
        min_depth=opt.min_depth,
        max_depth=opt.max_depth,
        use_median_scaling=use_scaling,
    )

    print("\n" + "═" * 80)
    print(f"  Model: {opt.checkpoint}")
    print(f"  Split: {opt.split_file}  ({len(all_preds)} frames)")
    print(f"  Median scaling: {'YES' if use_scaling else 'NO'}")
    print("─" * 80)
    print_metrics(mean_metrics, prefix="  RESULT")
    print("═" * 80)

    # Paper reference result for comparison
    print("\n  Paper (full KITTI, monocular, ImageNet pretrained):")
    print("  AbsRel=0.1150  SqRel=0.9030  RMSE=4.863  RMSElog=0.1930"
          "  δ<1.25=0.877  δ<1.25²=0.959  δ<1.25³=0.981")
    print("  (Expected to be significantly higher on a toy dataset)")

    if opt.output_json:
        with open(opt.output_json, "w") as f:
            json.dump(mean_metrics, f, indent=2)
        print(f"\n[Evaluate] Metrics saved → {opt.output_json}")


if __name__ == "__main__":
    main()
