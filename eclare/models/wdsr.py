"""
Adapted from https://github.com/JiahuiYu/wdsr_ntire2018/blob/master/wdsr_b.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from radifox.utils.resize.pytorch import resize
from math import ceil
import numpy as np


def pixel_shuffle(x, scale):
    """https://gist.github.com/davidaknowles/6e95a643adaf3960d1648a6b369e9d0b"""
    num_batches, num_channels, nx, ny = x.shape
    num_channels = num_channels // scale
    out = x.contiguous().view(num_batches, num_channels, scale, nx, ny)
    out = out.permute(0, 1, 3, 2, 4).contiguous()
    out = out.view(num_batches, num_channels, nx * scale, ny)
    return out


class Upsample(nn.Module):
    def __init__(self, num_channels, scale, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.scale = scale
        self.conv0 = nn.Conv2d(num_channels, scale, kernel_size, padding=padding)

    def forward(self, x):
        out = self.conv0(x)
        out = pixel_shuffle(out, self.scale)
        return out


class Block(nn.Module):
    def __init__(self, n_feats, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        expand = 4
        linear = 0.8
        self.body = nn.Sequential(
            *[
                nn.Conv2d(n_feats, n_feats * expand, 1, padding=0),
                act,
                nn.Conv2d(n_feats * expand, int(n_feats * linear), 1, padding=0),
                nn.Conv2d(int(n_feats * linear), n_feats, 3, padding=1),
            ]
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class WDSR(nn.Module):
    def __init__(self, n_resblocks, num_channels, scale, order=3, interp_wdsr=False):
        super().__init__()
        self.scale = scale
        self.order = order
        self.interp_wdsr = interp_wdsr

        kernel_size = 3
        padding = (kernel_size - 1) // 2
        act = nn.ReLU(True)

        self.head = nn.Conv2d(1, num_channels, kernel_size, padding=padding)

        self.body = nn.Sequential(
            *[Block(num_channels, act=act, res_scale=1) for _ in range(n_resblocks)]
        )

        if self.interp_wdsr:
            self.tail = nn.Conv1d(num_channels, 1, kernel_size, padding=padding)
        else:
            self.tail = Upsample(num_channels, ceil(self.scale), kernel_size)

    def calc_out_patch_size(self, input_patch_size):
        x = torch.rand([1, 1] + input_patch_size).float()
        x = x.to(next(self.parameters()).device)
        out, *_ = self(x)
        patch_size = list(out.shape[2:])

        return patch_size

    def forward(self, x):
        # Interpolate the input to create a skip
        s = resize(x, (1 / self.scale, 1), order=self.order)

        # Process in LR space
        x = self.head(x)
        x = self.body(x)

        if self.interp_wdsr:
            # Pixel shuffle to ceil(scale)
            x = self.tail(x)
            # Interpolate down to target shape
            x = resize(x, (1 / self.scale, 1), order=self.order)
        else:
            # Pixel shuffle to ceil(scale)
            x = self.tail(x)
            # Interpolate down to target shape
            x = resize(x, (ceil(self.scale) / self.scale, 1), order=self.order)

        # Track the residual
        r = x.clone()

        # Add the residual
        x = r + s

        # Clip to the range [0, 1]
        x = x.clip(0, 1)
        return x
