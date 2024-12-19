"""
This is the generator G for ESPRESO. It learns a PSF and also can apply the PSF.
"""

import torch
import math
from torch import nn
import torch.nn.functional as F

class ConvReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.body = nn.Sequential(
            *[
                nn.Conv2d(in_c, out_c, (kernel_size, 1)),
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        return self.body(x)

class G(nn.Module):    
    def __init__(self, num_channels=256, kernel_size=3, num_convs=2, length=21):
        super().__init__()
        
        size = length + (kernel_size - 1) * num_convs
        self.embedding = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, num_channels, size, 1), a=math.sqrt(5)))

        self.body = nn.Sequential(
            *[ConvReLU(num_channels, num_channels, kernel_size) for _ in range(num_convs - 1)],
            nn.Conv2d(num_channels, 1, (kernel_size, 1))
        )

    def forward(self):
        """
        Generate the PSF of shape 1 x 1 x length
        """
        psf = self.body(self.embedding)
        psf = self._symmetrize(psf)
        psf = F.softmax(psf, dim=2)
        
        return psf

    def _symmetrize(self, sp):
        return 0.5 * (sp + torch.flip(sp, (2, )))
