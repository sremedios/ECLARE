import numpy as np
import torch
from radifox.utils.degrade.degrade import select_kernel
from math import ceil


def parse_kernel(blur_kernel_file, blur_kernel_type, blur_fwhm):

    if blur_kernel_file is not None:
        blur_kernel = np.load(blur_kernel_file)
    else:
        window_size = int(2 * round(blur_fwhm) + 1)
        blur_kernel = select_kernel(window_size, blur_kernel_type, fwhm=blur_fwhm)
    blur_kernel /= blur_kernel.sum()
    blur_kernel = blur_kernel.squeeze()[None, None, :, None]
    blur_kernel = torch.from_numpy(blur_kernel).float()

    return blur_kernel


def calc_extended_patch_size(blur_kernel, patch_size):
    """
    Calculate the extended patch size. This is necessary to remove all boundary
    effects which could occur when we apply the blur kernel. We will pull a patch
    which is the specified patch size plus half the size of the blur kernel. Then we later
    blur at test time, crop off this extended patch size, then downsample.
    """

    L = blur_kernel.shape[0]

    ext_patch_size = [p + 2 * ceil(L / 2) if p != 1 else p for p in patch_size]
    ext_patch_crop = [(e - p) // 2 for e, p in zip(ext_patch_size, patch_size)]
    ext_patch_crop = tuple([slice(d, -d) for d in ext_patch_crop if d != 0])

    return ext_patch_size, ext_patch_crop
