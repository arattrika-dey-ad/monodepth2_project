# Monodepth2 From-Scratch Implementation

## Self-Supervised Monocular Depth Estimation

This is a complete from-scratch implementation of:
**"Digging Into Self-Supervised Monocular Depth Estimation"**
Godard et al., ICCV 2019 | https://arxiv.org/abs/1806.01260

---

## Project Structure

```
monodepth2_scratch/
├── README.md
├── requirements.txt
├── train.py                  # Main training script (your implementation)
├── evaluate.py               # Evaluation script
├── infer.py                  # Single-image inference / visualise depth
├── prepare_toy_dataset.py    # Script to extract a KITTI toy subset
├── models/
│   ├── __init__.py
│   ├── resnet_encoder.py     # ResNet-18 encoder (depth + pose)
│   ├── depth_decoder.py      # U-Net decoder, multi-scale depth
│   └── pose_decoder.py       # 6-DoF pose decoder
├── losses/
│   ├── __init__.py
│   ├── photometric.py        # SSIM + L1 photometric error
│   ├── reprojection.py       # Min-reprojection + auto-mask
│   └── smoothness.py         # Edge-aware smoothness loss
├── datasets/
│   ├── __init__.py
│   ├── kitti_dataset.py      # KITTI temporal triplet loader
│   └── transforms.py        # Augmentations (colour jitter, flip)
├── utils/
│   ├── __init__.py
│   ├── geometry.py           # Camera projection, backprojection, warping
│   └── visualise.py          # Depth colourmaps, tensorboard helpers
└── evaluation/
    ├── __init__.py
    └── metrics.py            # Abs Rel, Sq Rel, RMSE, RMSE log, δ thresholds
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate.bat       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download KITTI raw data (toy subset)

You need a small slice of KITTI raw data (~2 GB for a toy set).
Go to http://www.cvlibs.net/datasets/kitti/raw_data.php and download
**one drive sequence**, e.g.:

```
2011_09_26_drive_0001_sync  (a ~500-frame urban sequence)
```

Also download the calibration files for date `2011_09_26`.

Unzip to a folder, e.g.:

```
/data/kitti_raw/
  2011_09_26/
    calib_cam_to_cam.txt
    2011_09_26_drive_0001_sync/
      image_02/data/   ← left colour camera (we use this)
      image_03/data/   ← right colour camera
```

### 4. Prepare the toy dataset

```bash
python prepare_toy_dataset.py \
    --kitti_raw_path /data/kitti_raw \
    --output_path    /data/kitti_toy \
    --sequences      2011_09_26_drive_0001_sync \
    --max_frames     400
```

This creates the split files used by the data loader.

---

## Training your from-scratch model

```bash
python train.py \
    --data_path     /data/kitti_toy \
    --model_name    scratch_toy \
    --num_epochs    20 \
    --batch_size    4 \
    --height        192 \
    --width         640 \
    --learning_rate 1e-4 \
    --num_workers   4
```

Training logs are written to `logs/scratch_toy/`.
Checkpoints are saved to `checkpoints/scratch_toy/`.

---

## Running the official NianticLabs implementation

```bash
# Clone the official repo
git clone https://github.com/nianticlabs/monodepth2
cd monodepth2

# Install their dependencies (same torch, torchvision)
pip install -r requirements.txt

# Train on the same toy dataset
python train.py \
    --model_name official_toy \
    --data_path  /data/kitti_toy \
    --split      eigen_zhou \
    --num_epochs 20

# Evaluate
python evaluate_depth.py \
    --load_weights_folder ~/tmp/official_toy/models/weights_19/ \
    --data_path           /data/kitti_toy \
    --eval_split          eigen
```

---

## Evaluating your from-scratch model

```bash
python evaluate.py \
    --checkpoint   checkpoints/scratch_toy/epoch_19.pth \
    --data_path    /data/kitti_toy \
    --split_file   splits/toy_test.txt
```

---

## Inference / Visualisation

```bash
python infer.py \
    --checkpoint  checkpoints/scratch_toy/epoch_19.pth \
    --image       /path/to/any_image.jpg \
    --output_dir  output_depths/
```
