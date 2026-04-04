"""
infer.py
--------
Run a trained from-scratch Monodepth2 checkpoint on one or more images
and save coloured depth-map visualisations.

Usage
-----
# Single image
python infer.py \
    --checkpoint checkpoints/scratch_toy/epoch_19.pth \
    --image      /path/to/image.jpg \
    --output_dir output_depths/

# Folder of images
python infer.py \
    --checkpoint checkpoints/scratch_toy/epoch_19.pth \
    --image_dir  /path/to/images/ \
    --output_dir output_depths/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models import ResNetEncoder, DepthDecoder


def parse_args():
    p = argparse.ArgumentParser(description="Monodepth2 depth inference")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--image",       type=str, default=None,
                   help="Path to a single input image.")
    p.add_argument("--image_dir",   type=str, default=None,
                   help="Directory of images to process.")
    p.add_argument("--output_dir",  type=str, default="output_depths")
    p.add_argument("--height",      type=int, default=192)
    p.add_argument("--width",       type=int, default=640)
    p.add_argument("--ext",         type=str, default="png",
                   help="Output image extension.")
    return p.parse_args()


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    encoder = ResNetEncoder(num_layers=18, pretrained=False).to(device)
    decoder = DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=[0]
    ).to(device)

    encoder.load_state_dict(ckpt["depth_encoder"])
    decoder.load_state_dict(ckpt["depth_decoder"])

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def predict_depth(encoder, decoder,
                  img_pil: Image.Image,
                  height: int, width: int,
                  device) -> np.ndarray:
    """
    Returns a (H, W) depth map in metres for the given PIL image.
    """
    transform = transforms.Compose([
        transforms.Resize((height, width),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)   # (1, 3, H, W)

    with torch.no_grad():
        feats = encoder(img_tensor)
        out   = decoder(feats)
        disp  = out[("disp", 0)]   # (1, 1, H, W)

    # Convert disparity → depth
    min_d, max_d = 0.1, 100.0
    a = 1.0 / min_d - 1.0 / max_d
    b = 1.0 / max_d
    depth = (1.0 / (a * disp + b)).squeeze().cpu().numpy()   # (H, W)
    return depth


def save_depth_vis(depth: np.ndarray,
                   rgb:   np.ndarray,
                   out_path: Path):
    """Save a side-by-side RGB + coloured depth image."""
    # Normalise depth to [0,1] for colouring
    vmin = np.percentile(depth, 5)
    vmax = np.percentile(depth, 95)
    d_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colormap = cm.get_cmap("magma")
    depth_colour = (colormap(d_norm)[:, :, :3] * 255).astype(np.uint8)

    # Side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(depth_colour)
    axes[1].set_title("Predicted depth (magma, near=bright)", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    opt    = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder = load_model(opt.checkpoint, device)
    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    if opt.image:
        image_paths = [Path(opt.image)]
    elif opt.image_dir:
        d = Path(opt.image_dir)
        image_paths = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    else:
        raise ValueError("Provide --image or --image_dir")

    print(f"[Infer] Processing {len(image_paths)} image(s)...")

    for img_path in image_paths:
        pil_img = Image.open(img_path).convert("RGB")
        rgb_np  = np.array(pil_img)

        depth_np = predict_depth(
            encoder, decoder, pil_img, opt.height, opt.width, device
        )

        out_path = out_dir / (img_path.stem + f"_depth.{opt.ext}")
        save_depth_vis(depth_np, rgb_np, out_path)
        print(f"  Saved → {out_path}")

    print("[Infer] Done.")


if __name__ == "__main__":
    main()
