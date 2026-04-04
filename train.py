"""
train.py
--------
Main training loop for the from-scratch Monodepth2 implementation.

Orchestration (Phase 3 of the assignment):
  1.  Pass I_t through Depth Network → multi-scale depth maps D_t.
  2.  Pass (I_t, I_{t-1}) and (I_t, I_{t+1}) through Pose Network
      → transformations T_{t→t-1} and T_{t→t+1}.
  3.  Warp I_{t-1} and I_{t+1} into the viewpoint of I_t using
      predicted depth + pose + camera intrinsics (bilinear sampling).
  4.  Upsample all intermediate depth maps to input resolution (full-res
      multi-scale).
  5.  Compute combined loss:
        L = μ * L_p(min reprojection) + λ * L_s(smoothness)
      averaged over all 4 scales, all pixels, and the batch.
  6.  Back-propagate and update with Adam (lr=1e-4, then 1e-5).

Usage
-----
python train.py \
    --data_path     /data/kitti_toy \
    --model_name    scratch_toy \
    --num_epochs    20 \
    --batch_size    4 \
    --height        192 \
    --width         640 \
    --learning_rate 1e-4 \
    --num_workers   4
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Local imports ─────────────────────────────────────────────────
from models   import ResNetEncoder, DepthDecoder, PoseDecoder
from datasets import KITTIDataset
from losses   import compute_reprojection_loss, edge_aware_smoothness
from utils    import (BackprojectDepth, Project3D,
                      transformation_from_parameters)
from utils.visualise import tensor_to_depth_img


# ═════════════════════════════════════════════════════════════════
# Argument parsing
# ═════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Monodepth2 from-scratch training"
    )
    # Data
    p.add_argument("--data_path",   type=str, required=True)
    p.add_argument("--model_name",  type=str, default="scratch_model")
    p.add_argument("--split",       type=str, default="toy",
                   help="Prefix of split files in data_path/splits/")

    # Architecture
    p.add_argument("--height",      type=int, default=192)
    p.add_argument("--width",       type=int, default=640)
    p.add_argument("--scales",      nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--pretrained",  action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")

    # Training
    p.add_argument("--num_epochs",       type=int,   default=20)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--learning_rate",    type=float, default=1e-4)
    p.add_argument("--scheduler_step",   type=int,   default=15,
                   help="Epoch at which lr drops from 1e-4 to 1e-5")
    p.add_argument("--smoothness_weight",type=float, default=1e-3,
                   help="Lambda for smoothness loss (paper: 0.001)")
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--log_freq",         type=int,   default=100,
                   help="Log to TensorBoard every N steps")
    p.add_argument("--save_freq",        type=int,   default=5,
                   help="Save checkpoint every N epochs")

    # Paths
    p.add_argument("--log_dir",   type=str, default="logs")
    p.add_argument("--ckpt_dir",  type=str, default="checkpoints")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════
# Trainer class
# ═════════════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Trainer] Using device: {self.device}")

        # ── Output directories ────────────────────────────────────────
        self.log_path  = Path(opt.log_dir)  / opt.model_name
        self.ckpt_path = Path(opt.ckpt_dir) / opt.model_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.ckpt_path.mkdir(parents=True, exist_ok=True)

        # ── Save options ──────────────────────────────────────────────
        with open(self.ckpt_path / "opt.json", "w") as f:
            json.dump(vars(opt), f, indent=2)

        # ── Build models ──────────────────────────────────────────────
        self._build_models()

        # ── Geometry helpers (batch-size dependent) ───────────────────
        self.backproject = BackprojectDepth(
            opt.batch_size, opt.height, opt.width
        ).to(self.device)
        self.project3d   = Project3D(
            opt.batch_size, opt.height, opt.width
        ).to(self.device)

        # ── Data loaders ──────────────────────────────────────────────
        self._build_dataloaders()

        # ── Optimiser + scheduler ─────────────────────────────────────
        all_params = (
            list(self.depth_encoder.parameters()) +
            list(self.depth_decoder.parameters()) +
            list(self.pose_encoder.parameters())  +
            list(self.pose_decoder.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=opt.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=opt.scheduler_step,
            gamma=0.1            # 1e-4 → 1e-5 at epoch 15
        )

        # ── TensorBoard ───────────────────────────────────────────────
        self.writer     = SummaryWriter(self.log_path)
        self.global_step = 0

        print(f"[Trainer] Training for {opt.num_epochs} epochs")
        print(f"[Trainer] Dataset size: {len(self.train_loader.dataset)} samples")

    # ─────────────────────────────────────────────────────────────
    # Model construction
    # ─────────────────────────────────────────────────────────────

    def _build_models(self):
        opt = self.opt

        # Depth encoder: ResNet-18, 3-channel, ImageNet pretrained
        self.depth_encoder = ResNetEncoder(
            num_layers=18,
            pretrained=opt.pretrained,
            num_input_images=1
        ).to(self.device)

        # Depth decoder: U-Net, 4 output scales
        self.depth_decoder = DepthDecoder(
            num_ch_enc=self.depth_encoder.num_ch_enc,
            scales=opt.scales
        ).to(self.device)

        # Pose encoder: ResNet-18, 6-channel (two frames stacked)
        self.pose_encoder = ResNetEncoder(
            num_layers=18,
            pretrained=opt.pretrained,
            num_input_images=2
        ).to(self.device)

        # Pose decoder: 6-DoF
        self.pose_decoder = PoseDecoder(
            num_ch_enc=self.pose_encoder.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=1
        ).to(self.device)

        print("[Models] Depth encoder params: "
              f"{sum(p.numel() for p in self.depth_encoder.parameters()):,}")
        print("[Models] Total trainable params: "
              f"{sum(p.numel() for p in self.depth_encoder.parameters()) + sum(p.numel() for p in self.depth_decoder.parameters()) + sum(p.numel() for p in self.pose_encoder.parameters()) + sum(p.numel() for p in self.pose_decoder.parameters()):,}")

    # ─────────────────────────────────────────────────────────────
    # Dataloaders
    # ─────────────────────────────────────────────────────────────

    def _build_dataloaders(self):
        opt = self.opt
        split_dir = Path(opt.data_path) / "splits"

        train_dataset = KITTIDataset(
            data_path=opt.data_path,
            split_file=str(split_dir / f"{opt.split}_train.txt"),
            height=opt.height,
            width=opt.width,
            is_train=True,
            frame_idxs=[0, -1, 1],
        )
        val_dataset = KITTIDataset(
            data_path=opt.data_path,
            split_file=str(split_dir / f"{opt.split}_val.txt"),
            height=opt.height,
            width=opt.width,
            is_train=False,
            frame_idxs=[0, -1, 1],
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ─────────────────────────────────────────────────────────────
    # Main training entry point
    # ─────────────────────────────────────────────────────────────

    def train(self):
        for epoch in range(self.opt.num_epochs):
            t0 = time.time()
            train_loss = self._run_epoch(epoch, is_train=True)
            val_loss   = self._run_epoch(epoch, is_train=False)
            self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1:02d}/{self.opt.num_epochs}]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"lr={lr:.2e}  time={elapsed:.1f}s"
            )
            self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            self.writer.add_scalar("Loss/val_epoch",   val_loss,   epoch)

            if (epoch + 1) % self.opt.save_freq == 0 or \
               epoch == self.opt.num_epochs - 1:
                self._save_checkpoint(epoch)

        self.writer.close()
        print("[Trainer] Training complete.")

    # ─────────────────────────────────────────────────────────────
    # One epoch
    # ─────────────────────────────────────────────────────────────

    def _run_epoch(self, epoch: int, is_train: bool) -> float:
        loader = self.train_loader if is_train else self.val_loader
        if is_train:
            self.depth_encoder.train()
            self.depth_decoder.train()
            self.pose_encoder.train()
            self.pose_decoder.train()
        else:
            self.depth_encoder.eval()
            self.depth_decoder.eval()
            self.pose_encoder.eval()
            self.pose_decoder.eval()

        total_loss = 0.0
        ctx = torch.enable_grad() if is_train else torch.no_grad()

        with ctx:
            for batch_idx, inputs in enumerate(loader):
                loss, losses_dict = self._process_batch(inputs)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.global_step % self.opt.log_freq == 0:
                        self._log_step(losses_dict, inputs)
                    self.global_step += 1

                total_loss += loss.item()

        return total_loss / max(1, len(loader))

    # ─────────────────────────────────────────────────────────────
    # Process one batch — core forward pass
    # ─────────────────────────────────────────────────────────────

    def _process_batch(self, inputs: dict):
        """
        Full forward pass for one batch.

        Steps
        -----
        1.  Move inputs to device.
        2.  Run depth network on I_t (augmented).
        3.  Run pose network for each source frame pair.
        4.  For each scale:
              a.  Upsample disparity to full (H, W).
              b.  Back-project depth → 3D points.
              c.  Transform points using predicted pose.
              d.  Project → normalised sampling coords.
              e.  Bilinearly sample source images.
              f.  Compute min-reprojection + auto-mask loss.
              g.  Compute smoothness loss.
        5.  Accumulate loss across scales.
        6.  Return total loss scalar.
        """
        opt = self.opt

        # ── Move to device ────────────────────────────────────────────
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        target_img     = inputs[("color",     0, 0)]   # for loss
        target_img_aug = inputs[("color_aug", 0, 0)]   # fed to depth net

        source_imgs = [
            inputs[("color", -1, 0)],
            inputs[("color",  1, 0)],
        ]
        source_imgs_aug = [
            inputs[("color_aug", -1, 0)],
            inputs[("color_aug",  1, 0)],
        ]

        K    = inputs["K"]       # (B, 4, 4)
        invK = inputs["inv_K"]   # (B, 4, 4)

        # ── Step 1: Depth network ─────────────────────────────────────
        enc_feats  = self.depth_encoder(target_img_aug)
        depth_outs = self.depth_decoder(enc_feats)
        # depth_outs[("disp", s)] for s in {0,1,2,3}

        # ── Step 2: Pose network ─────────────────────────────────────
        # For each source frame, concatenate with target and run encoder
        poses = {}
        for fi, src_aug in zip([-1, 1], source_imgs_aug):
            # Stack [I_t, I_{t'}] along channel dim → (B, 6, H, W)
            pose_input = torch.cat([target_img_aug, src_aug], dim=1)
            pose_feats = self.pose_encoder(pose_input)

            axisangle, translation = self.pose_decoder([pose_feats])
            # axisangle:   (B, 1, 1, 3)
            # translation: (B, 1, 1, 3)

            # Build 4×4 transformation matrix T_{t → t'}
            T = transformation_from_parameters(
                axisangle[:, 0],        # (B, 1, 3)
                translation[:, 0],      # (B, 1, 3)
                invert=(fi < 0)         # invert for t→t-1
            )
            poses[fi] = T

        # ── Steps 3–5: Warp + loss at each scale ─────────────────────
        total_loss   = 0.0
        losses_dict  = {}

        for scale in opt.scales:
            disp = depth_outs[("disp", scale)]   # (B, 1, H_s, W_s)

            # ── Full-resolution multi-scale (Section 3.2, Fig 3d) ─────
            # Upsample disparity to the ORIGINAL input resolution
            # BEFORE computing the photometric loss.
            disp_full = F.interpolate(
                disp,
                size=(opt.height, opt.width),
                mode="bilinear",
                align_corners=False
            )                                    # (B, 1, H, W)

            # Convert disparity → depth
            min_depth, max_depth = 0.1, 100.0
            a = 1.0 / min_depth - 1.0 / max_depth
            b = 1.0 / max_depth
            depth_full = 1.0 / (a * disp_full + b)   # (B, 1, H, W)

            # ── Back-project to 3D ────────────────────────────────────
            cam_points = self.backproject(depth_full, invK)   # (B, 4, H*W)

            # ── Warp each source frame ────────────────────────────────
            warped_imgs = []
            for fi, src in zip([-1, 1], source_imgs):
                T = poses[fi]

                # Project 3D points into source image plane
                pix_coords = self.project3d(cam_points, K, T)
                # pix_coords : (B, H, W, 2) in [-1, 1]

                # Bilinear sample the source image
                warped = F.grid_sample(
                    src,
                    pix_coords,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True
                )                        # (B, 3, H, W)
                warped_imgs.append(warped)

            # ── Min-reprojection + auto-mask loss ─────────────────────
            reproj_loss, mask, min_err = compute_reprojection_loss(
                warped_imgs=warped_imgs,
                target_img=target_img,
                source_imgs=source_imgs,
                automasking=True
            )

            # ── Smoothness loss ────────────────────────────────────────
            smooth_loss = edge_aware_smoothness(disp_full, target_img)

            # ── Combine ───────────────────────────────────────────────
            scale_loss = reproj_loss + opt.smoothness_weight * smooth_loss
            total_loss += scale_loss

            losses_dict[f"reproj/s{scale}"]  = reproj_loss.item()
            losses_dict[f"smooth/s{scale}"]  = smooth_loss.item()

        total_loss /= len(opt.scales)   # mean over scales
        losses_dict["total"] = total_loss.item()

        # Store some tensors for visualisation
        losses_dict["_disp0"]     = depth_outs[("disp", 0)].detach()
        losses_dict["_mask"]      = mask.detach()
        losses_dict["_target"]    = target_img.detach()
        losses_dict["_warped"]    = warped_imgs[0].detach()

        return total_loss, losses_dict

    # ─────────────────────────────────────────────────────────────
    # TensorBoard logging
    # ─────────────────────────────────────────────────────────────

    def _log_step(self, losses_dict: dict, inputs: dict):
        step = self.global_step
        for k, v in losses_dict.items():
            if not k.startswith("_") and isinstance(v, float):
                self.writer.add_scalar(f"Loss/{k}", v, step)

        # Log a depth image from the first sample in the batch
        if "_disp0" in losses_dict:
            disp_img = tensor_to_depth_img(losses_dict["_disp0"][:1])
            self.writer.add_image(
                "Depth/disp_scale0",
                disp_img.transpose(2, 0, 1),   # (C, H, W)
                step
            )

        # Log auto-mask — squeeze to (1, H, W) for TensorBoard
        if "_mask" in losses_dict:
            mask_img = losses_dict["_mask"][:1].squeeze(0).cpu().float()
            self.writer.add_image("Mask/automask", mask_img, step)

    # ─────────────────────────────────────────────────────────────
    # Checkpoint save / load
    # ─────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int):
        ckpt = {
            "epoch":          epoch,
            "depth_encoder":  self.depth_encoder.state_dict(),
            "depth_decoder":  self.depth_decoder.state_dict(),
            "pose_encoder":   self.pose_encoder.state_dict(),
            "pose_decoder":   self.pose_decoder.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict(),
        }
        path = self.ckpt_path / f"epoch_{epoch:02d}.pth"
        torch.save(ckpt, path)
        print(f"[Checkpoint] Saved → {path}")

    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.depth_encoder.load_state_dict(ckpt["depth_encoder"])
        self.depth_decoder.load_state_dict(ckpt["depth_decoder"])
        self.pose_encoder.load_state_dict(ckpt["pose_encoder"])
        self.pose_decoder.load_state_dict(ckpt["pose_decoder"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[Checkpoint] Loaded ← {ckpt_path}")


# ═════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    opt = parse_args()
    trainer = Trainer(opt)
    trainer.train()
