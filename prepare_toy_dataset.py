"""
prepare_toy_dataset.py
----------------------
Extracts a small "toy" subset of the KITTI raw dataset and creates the
split files (train/val/test) that KITTIDataset expects.

Usage
-----
python prepare_toy_dataset.py \
    --kitti_raw_path /data/kitti_raw \
    --output_path    /data/kitti_toy \
    --sequences      2011_09_26_drive_0001_sync \
    --max_frames     400

What this script does
---------------------
1.  Scans each sequence folder for image_02/data/*.png frames.
2.  Skips the first and last frame (no temporal triplet possible).
3.  Shuffles and splits into 70% train / 15% val / 15% test.
4.  Writes split files:
      <output_path>/splits/toy_train.txt
      <output_path>/splits/toy_val.txt
      <output_path>/splits/toy_test.txt
    Each line: "<sequence_folder> <frame_index>"
5.  Copies (or symlinks) the image data into <output_path>.

Note: The split format follows the convention of the official
monodepth2 repo so the same split files work with both implementations.
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare KITTI toy dataset for Monodepth2"
    )
    p.add_argument("--kitti_raw_path", type=str, required=True,
                   help="Root of the downloaded KITTI raw data.")
    p.add_argument("--output_path",    type=str, required=True,
                   help="Where to write the toy dataset.")
    p.add_argument("--sequences",      nargs="+", required=True,
                   help="KITTI drive sequence folder names, e.g. "
                        "2011_09_26_drive_0001_sync")
    p.add_argument("--max_frames",     type=int, default=400,
                   help="Maximum number of triplets to use (total).")
    p.add_argument("--train_frac",     type=float, default=0.70)
    p.add_argument("--val_frac",       type=float, default=0.15)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--symlink",        action="store_true",
                   help="Symlink instead of copying image files.")
    return p.parse_args()


def collect_triplets(kitti_raw: Path, sequences: list) -> list:
    """
    Return a list of (seq_name, frame_idx) for every frame that has
    both a previous and a next frame (i.e. not the very first/last).
    """
    triplets = []
    for seq in sequences:
        img_dir = kitti_raw / seq / "image_02" / "data"
        if not img_dir.exists():
            print(f"  [WARNING] Not found: {img_dir}")
            continue

        frames = sorted(img_dir.glob("*.png"))
        if not frames:
            frames = sorted(img_dir.glob("*.jpg"))

        # Parse frame indices from filenames
        indices = []
        for f in frames:
            try:
                indices.append(int(f.stem))
            except ValueError:
                pass
        indices.sort()

        # Only keep frames where both i-1 and i+1 exist
        index_set = set(indices)
        for idx in indices:
            if (idx - 1) in index_set and (idx + 1) in index_set:
                triplets.append((seq, idx))

        print(f"  Sequence {seq}: {len(indices)} frames → "
              f"{len([x for x in triplets if x[0]==seq])} valid triplets")

    return triplets


def main():
    opt = parse_args()
    random.seed(opt.seed)

    raw_path = Path(opt.kitti_raw_path)
    out_path = Path(opt.output_path)

    print(f"[Prepare] KITTI raw:  {raw_path}")
    print(f"[Prepare] Output:     {out_path}")
    print(f"[Prepare] Sequences:  {opt.sequences}")

    # ── 1. Collect all valid triplets ─────────────────────────────
    print("\n[Prepare] Scanning sequences...")
    all_triplets = collect_triplets(raw_path, opt.sequences)
    print(f"[Prepare] Total valid triplets: {len(all_triplets)}")

    if len(all_triplets) == 0:
        print("[ERROR] No valid triplets found. Check --kitti_raw_path and --sequences.")
        sys.exit(1)

    # ── 2. Subsample to max_frames ────────────────────────────────
    if len(all_triplets) > opt.max_frames:
        all_triplets = random.sample(all_triplets, opt.max_frames)
        all_triplets.sort()
    print(f"[Prepare] Using {len(all_triplets)} triplets.")

    # ── 3. Train / val / test split ───────────────────────────────
    random.shuffle(all_triplets)
    n = len(all_triplets)
    n_train = int(n * opt.train_frac)
    n_val   = int(n * opt.val_frac)
    n_test  = n - n_train - n_val

    train_set = all_triplets[:n_train]
    val_set   = all_triplets[n_train : n_train + n_val]
    test_set  = all_triplets[n_train + n_val:]

    print(f"[Prepare] Split → train:{len(train_set)}  "
          f"val:{len(val_set)}  test:{len(test_set)}")

    # ── 4. Copy / symlink image data ──────────────────────────────
    print("\n[Prepare] Copying image files (this may take a moment)...")
    copied = 0
    for seq, idx in all_triplets:
        for offset in [-1, 0, 1]:
            frame = idx + offset
            src_png = raw_path / seq / "image_02" / "data" / f"{frame:010d}.png"
            src_jpg = raw_path / seq / "image_02" / "data" / f"{frame:010d}.jpg"
            src     = src_png if src_png.exists() else src_jpg
            if not src.exists():
                continue

            dst_dir = out_path / seq / "image_02" / "data"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name

            if dst.exists():
                continue

            if opt.symlink:
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
            copied += 1

    print(f"[Prepare] Copied {copied} image files.")

    # ── 5. Write split files ──────────────────────────────────────
    splits_dir = out_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write_split(entries, filename):
        path = splits_dir / filename
        with open(path, "w") as f:
            for seq, idx in entries:
                f.write(f"{seq} {idx}\n")
        print(f"[Prepare] Wrote {filename} ({len(entries)} entries) → {path}")

    write_split(train_set, "toy_train.txt")
    write_split(val_set,   "toy_val.txt")
    write_split(test_set,  "toy_test.txt")

    # ── 6. Summary ────────────────────────────────────────────────
    print("\n[Prepare] Done. Toy dataset ready at:", out_path)
    print("\nNext step — train your model:")
    print(f"  python train.py --data_path {out_path} --model_name scratch_toy --num_epochs 20")


if __name__ == "__main__":
    main()
