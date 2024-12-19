"""
From https://github.com/JiahuiYu/wdsr_ntire2018/blob/master/wdsr_b.py

Adapted for rational sampling rates, slight refactor for form.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from radifox.utils.resize.pytorch import resize
import numpy as np

# Handle breaking library change between torch versions
try:
    import torch.nn.utils.parametrizations.weight_norm as weight_norm
except:
    import torch.nn.utils.weight_norm as weight_norm

def pixel_shuffle(x, scale):
    """https://gist.github.com/davidaknowles/6e95a643adaf3960d1648a6b369e9d0b"""
    num_batches, num_channels, nx, ny = x.shape
    num_channels = num_channels // scale
    out = x.contiguous().view(num_batches, num_channels, scale, nx, ny)
    out = out.permute(0, 1, 3, 2, 4).contiguous()
    out = out.view(num_batches, num_channels, nx * scale, ny)
    return out


class Upsample(nn.Module):
    def __init__(self, num_channels, scale, kernel_size, wn):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.scale = scale
        self.conv0 = wn(nn.Conv2d(num_channels, scale, kernel_size, padding=padding))

    def forward(self, x):
        out = self.conv0(x)
        out = pixel_shuffle(out, self.scale)
        return out


class Block(nn.Module):
    def __init__(self, n_feats, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        expand = 4
        linear = 0.8
        self.body = nn.Sequential(
            *[
                wn(nn.Conv2d(n_feats, n_feats * expand, 1, padding=0)),
                act,
                wn(nn.Conv2d(n_feats * expand, int(n_feats * linear), 1, padding=0)),
                wn(nn.Conv2d(int(n_feats * linear), n_feats, 3, padding=1)),
            ]
        )

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class WDSR(nn.Module):
    def __init__(self, n_resblocks, num_channels, scale, order=3):
        super().__init__()
        self.scale = scale
        self.order = order
        self._scale1 = int(scale)
        self._scale0 = scale / float(self._scale1)

        wn = lambda x: weight_norm(x)
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        act = nn.ReLU(True)

        self.head = wn(nn.Conv2d(1, num_channels, kernel_size, padding=padding))

        self.body = nn.Sequential(
            *[
                Block(num_channels, act=act, res_scale=1, wn=wn)
                for _ in range(n_resblocks)
            ]
        )

        self.tail = Upsample(num_channels, self._scale1, kernel_size, wn)
        self.skip = Upsample(1, self._scale1, 5, wn)

    def calc_out_patch_size(self, input_patch_size):
        x = torch.rand([1, 1] + input_patch_size).float()
        x = x.to(next(self.parameters()).device)
        out, *_ = self(x)
        patch_size = list(out.shape[2:])
        return patch_size

    def forward(self, x):
        x = resize(x, (1 / self._scale0, 1), order=self.order)
        s = self.skip(x)

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        r = x.clone()
        x += s
        return x, r, s
