import numpy as np
import torch


def rotate_vol_2d(vol, angle):
    """Rotates each 2D slice in a volume by an angle

    Inputs:
        vol: torch Tensor of shape (hr_axis, hr_axis, lr_axis)
        angle: angle to rotate
    Outputs:
        Rotated volume of shape (hr_axis, hr_axis, lr_axis) where rotation occured
        only in the hr-plane.
    """
    # Avoid interpolation errors when able to rot90
    if angle == 0 or angle == 360:
        return vol
    elif angle == 90:
        return torch.rot90(vol, k=1, dims=[0, 1])
    elif angle == -90:
        return torch.rot90(vol, k=-1, dims=[0, 1])
    elif angle == 180:
        return torch.rot90(vol, k=2, dims=[0, 1])
    elif angle == -180:
        return torch.rot90(vol, k=-2, dims=[0, 1])
    elif angle == 270:
        return torch.rot90(vol, k=3, dims=[0, 1])
    elif angle == -270:
        return torch.rot90(vol, k=-3, dims=[0, 1])
    else:
        raise NotImplementedError("Angles other than 90 degree rotations are not supported.")
