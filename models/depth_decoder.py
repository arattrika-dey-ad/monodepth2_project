"""
depth_decoder.py
----------------
U-Net style depth decoder.

Key paper details (Section 3.3 + Table 5 in supplementary):
  • Skip connections from all 5 encoder levels.
  • ELU activations throughout (NOT ReLU).
  • Reflection padding instead of zero-padding to reduce border artefacts.
  • Sigmoid at each output layer → converted to depth:
        D = 1 / (a * σ + b)
    where a, b are chosen so D ∈ [min_depth=0.1, max_depth=100].
  • 4 disparity outputs (one per decoder level), all UPSAMPLED to
    input resolution before the loss is computed (full-res multi-scale).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """
    Conv2d (with reflection padding) → ELU.
    Uses reflection padding to suppress border artefacts.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=0   # padding applied manually
        )
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reflection-pad by 1 on all sides before the 3×3 conv
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x = self.conv(x)
        return self.nonlin(x)


class DepthDecoder(nn.Module):
    """
    Decodes the 5-level ResNet-18 feature pyramid into multi-scale
    disparity maps.

    Args
    ----
    num_ch_enc   : array of encoder channel counts (from ResNetEncoder).
    scales       : list of output scales, e.g. [0, 1, 2, 3].
                   Scale 0 = finest (full resolution ÷ 1),
                   scale 3 = coarsest (÷ 8 before upsampling).
    num_output_channels : 1 for disparity.
    min_depth, max_depth : depth clipping values (paper: 0.1 and 100).
    """

    def __init__(self,
                 num_ch_enc: np.ndarray,
                 scales: list = [0, 1, 2, 3],
                 num_output_channels: int = 1,
                 min_depth: float = 0.1,
                 max_depth: float = 100.0):
        super().__init__()

        self.scales = scales
        self.num_output_channels = num_output_channels
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Decoder channel widths (from supplementary Table 5)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # ── Build upconv + iconv for each level ───────────────────────────
        # Naming follows official repo: upconv{scale}_{0|1}
        #   upconv{s}_0 : upsample + reduce channels
        #   upconv{s}_1 : fuse with skip, refine
        self.convs = nn.ModuleDict()

        for i in range(4, -1, -1):
            # upconv 0 : input is previous decoder level output
            num_ch_in = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}_0"] = ConvBlock(num_ch_in, num_ch_out)

            # upconv 1 : input is upconv0 + skip connection
            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_in += num_ch_enc[i - 1]
            self.convs[f"upconv_{i}_1"] = ConvBlock(num_ch_in, num_ch_out)

        # ── Disparity output heads ────────────────────────────────────────
        for s in self.scales:
            self.convs[f"dispconv_{s}"] = nn.Conv2d(
                self.num_ch_dec[s], num_output_channels,
                kernel_size=3, padding=1
            )

        self.sigmoid = nn.Sigmoid()

        # Constants for depth conversion: D = 1 / (a*σ + b)
        self.a = 1.0 / min_depth - 1.0 / max_depth   # ≈ 9.9
        self.b = 1.0 / max_depth                      # = 0.01

    def forward(self, input_features: list):
        """
        Args
        ----
        input_features : list of 5 tensors from the encoder,
                         ordered coarse → fine (index 0 = finest).

        Returns
        -------
        outputs : dict with keys ("disp", scale) → (B, 1, H_s, W_s)
                  AND ("depth", scale) → same shape in metres.
                  H_s, W_s are the spatial size at each decoder level
                  BEFORE upsampling to full resolution.
        """
        outputs = {}
        x = input_features[-1]      # start from the coarsest feature

        for i in range(4, -1, -1):
            x = self.convs[f"upconv_{i}_0"](x)
            x = [F.interpolate(x, scale_factor=2, mode='nearest')]

            if i > 0:
                # Concatenate skip connection from encoder
                x += [input_features[i - 1]]

            x = torch.cat(x, dim=1)
            x = self.convs[f"upconv_{i}_1"](x)

            if i in self.scales:
                disp = self.sigmoid(self.convs[f"dispconv_{i}"](x))
                outputs[("disp", i)] = disp

                # Convert sigmoid disparity to depth
                depth = 1.0 / (self.a * disp + self.b)
                outputs[("depth", i)] = depth

        return outputs
