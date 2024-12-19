"""
This is the discriminator D for ESPRESO.

TODO: recode
"""

import torch
import math
from torch import nn
import torch.nn.utils.spectral_norm as sn
import torch.nn.functional as F

class ConvSpectnormLReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, alpha):
        super().__init__()
        self.body = nn.Sequential(
            *[
                sn(nn.Conv2d(in_c, out_c, kernel_size)),
                nn.LeakyReLU(alpha),                
            ]
        )

    def forward(self, x):
        return self.body(x)

class D(nn.Module):
    def __init__(self, 
                 inp_ch=1,
                 out_chs=(64, 64, 64, 64),
                 kernel_sizes=((3, 1), (3, 1), (3, 1), (3, 1), (3, 1)),
                 alpha=0.1):

        
        super().__init__()
        assert len(out_chs) == len(kernel_sizes) - 1

        inp_cs = [inp_ch, *out_chs[:-1]]
        out_cs = out_chs[1:]

        self.body = nn.Sequential(
            *[
                ConvSpectnormLReLU(in_c, out_c, kernel_size, alpha)
                for in_c, out_c, kernel_size in zip(inp_cs, out_cs, kernel_sizes)
            ],
            sn(nn.Conv2d(out_chs[-1], 1, kernel_sizes[-1])),
        )
            
    def forward(self, x):
        return self.body(x)