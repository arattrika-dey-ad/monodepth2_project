"""
pose_decoder.py
---------------
Simple convolutional pose decoder on top of the ResNet-18 pose encoder.

Key paper details (Section 3.3 + supplementary Table 5):
  • Takes the final encoder feature (B, 512, H/32, W/32).
  • Three 1×1 or 3×3 convolutions → a (B, 6, 1, 1) output.
  • Interprets the 6 values as [rx, ry, rz, tx, ty, tz]:
        rotation    : axis-angle representation
        translation : (tx, ty, tz)
  • Both rotation AND translation are scaled by 0.01 to keep them
    in a numerically stable range during early training.
  • The decoder outputs ONE relative pose per frame pair; the
    encoder is called separately for each pair.
"""

import torch
import torch.nn as nn


class PoseDecoder(nn.Module):
    """
    Args
    ----
    num_ch_enc : list / array of encoder channel widths.
                 We only use the last value (512 for ResNet-18).
    num_input_features : how many encoder outputs are concatenated
                         (always 1 in our two-frame setting).
    num_frames_to_predict_for : how many poses to predict
                                (always 1: t → t±1).
    stride : stride of the final conv (1).
    """

    def __init__(self,
                 num_ch_enc,
                 num_input_features: int = 1,
                 num_frames_to_predict_for: int = 1,
                 stride: int = 1):
        super().__init__()

        self.num_frames_to_predict_for = num_frames_to_predict_for

        # Channel width entering the decoder = last encoder width
        num_ch_in = num_ch_enc[-1] * num_input_features

        # Three conv layers (paper supplementary Table 5)
        self.squeezeconv = nn.Conv2d(num_ch_in,  256, kernel_size=1)
        self.poseconv1   = nn.Conv2d(256, 256, kernel_size=3, stride=stride, padding=1)
        self.poseconv2   = nn.Conv2d(256, 256, kernel_size=3, stride=stride, padding=1)
        self.poseconv3   = nn.Conv2d(
            256,
            6 * num_frames_to_predict_for,
            kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_features: list):
        """
        Args
        ----
        input_features : list containing the *last* encoder feature map,
                         shape (B, 512, H/32, W/32).

        Returns
        -------
        axisangle  : (B, num_frames, 1, 3) rotation in axis-angle form
        translation: (B, num_frames, 1, 3)

        Both are scaled by 0.01 to ensure small initial values.
        """
        # Use the deepest encoder feature only
        last_features = [f[-1] for f in input_features]
        cat_features = torch.cat(last_features, dim=1)

        x = self.relu(self.squeezeconv(cat_features))
        x = self.relu(self.poseconv1(x))
        x = self.relu(self.poseconv2(x))
        out = self.poseconv3(x)

        # Global average pool to get a single vector per sample
        out = out.mean(dim=[2, 3])          # (B, 6 * num_frames)

        # Scale both rotation and translation by 0.01
        out = 0.01 * out

        # Reshape to (B, num_frames, 1, 6) then split
        out = out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle   = out[..., :3]   # (B, num_frames, 1, 3)
        translation = out[..., 3:]   # (B, num_frames, 1, 3)

        return axisangle, translation
