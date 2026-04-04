"""
resnet_encoder.py
-----------------
ResNet-18 encoder used by both the Depth network (3-channel input)
and the Pose network (6-channel input — two stacked RGB frames).

Key paper details (Section 3.3):
  • Depth encoder: standard ResNet18, ImageNet pretrained weights.
  • Pose encoder : same ResNet18 but first conv layer modified to
    accept 6 channels. Pretrained weights are DUPLICATED along the
    channel axis and divided by 2 so the output magnitude is the
    same as the single-image version.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    Wraps a torchvision ResNet-18.

    Args
    ----
    num_layers   : int  – only 18 is used in the paper.
    pretrained   : bool – load ImageNet weights.
    num_input_images : int
        1  → standard depth encoder (3-channel conv1)
        2  → pose encoder (6-channel conv1, weights duplicated & /2)
    """

    def __init__(self, num_layers: int = 18,
                 pretrained: bool = True,
                 num_input_images: int = 1):
        super().__init__()

        # ── 1. Build the base ResNet ──────────────────────────────────────
        assert num_layers == 18, "Only ResNet-18 is used in Monodepth2."
        self.encoder = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # ── 2. Modify conv1 for multi-image (pose) input ─────────────────
        if num_input_images > 1:
            # Original conv1: (3, 64, 7, 7)
            w = self.encoder.conv1.weight.data  # shape (64, 3, 7, 7)
            # Tile along the input-channel axis: (64, 6, 7, 7)
            w = w.repeat(1, num_input_images, 1, 1)
            # Divide by num_input_images so the output magnitude is
            # the same as for a single image (see supplementary B).
            w = w / float(num_input_images)

            new_conv1 = nn.Conv2d(
                in_channels=3 * num_input_images,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            new_conv1.weight = nn.Parameter(w)
            self.encoder.conv1 = new_conv1

        # ── 3. Record the number of channels at each skip-connection ──────
        # These are the channel counts fed to the depth decoder via skips.
        # ResNet-18 feature map channels: 64, 64, 128, 256, 512
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

    def forward(self, x: torch.Tensor):
        """
        Returns a list of 5 feature maps at decreasing spatial resolutions:
          features[0] : after conv1+bn+relu      (B, 64,  H/2,  W/2)
          features[1] : after layer1             (B, 64,  H/4,  W/4)
          features[2] : after layer2             (B, 128, H/8,  W/8)
          features[3] : after layer3             (B, 256, H/16, W/16)
          features[4] : after layer4             (B, 512, H/32, W/32)
        """
        features = []

        # stem
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)          # stride 2  → H/2

        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        features.append(x)          # stride 4  → H/4

        x = self.encoder.layer2(x)
        features.append(x)          # stride 8  → H/8

        x = self.encoder.layer3(x)
        features.append(x)          # stride 16 → H/16

        x = self.encoder.layer4(x)
        features.append(x)          # stride 32 → H/32

        return features
