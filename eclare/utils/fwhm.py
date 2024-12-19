import torch
import numpy as np
from scipy.interpolate import interp1d


def calc_fwhm(kernel):
    """Calculates the full width at half maximum (FWHM) using linear interp.

    Args:
        kernel (numpy.ndarray): The kernel whose FWHM to be calculated.

    Returns
    -------
    fwhm: float
        The calculated FWHM. It is equal to ``right - left``.
    left: float
        The position of the left of the FWHM.
    right: float
        The position of the right of the FWHM.

    """
    kernel = kernel.squeeze()
    half_max = float(np.max(kernel)) / 2
    indices = np.where(kernel > half_max)[0] 
    if len(indices) == 0:
        return 0, 0, 0

    left = indices[0]
    if left > 0:
        interp = interp1d((kernel[left-1], kernel[left]), (left - 1, left))
        left = interp(half_max)
    right = indices[-1]
    if right < len(kernel) - 1:
        interp = interp1d((kernel[right+1], kernel[right]), (right + 1, right))
        right = interp(half_max)
    fwhm = right - left
    return fwhm, left, right


def calc_fwhm_torch(kernel, return_left_right=False):
    """Calculates FWHM using linear interp for pytorch.

    Args:
        kernel (torch.Tensor): The kernel whose FWHM to be calculated.
        return_left_right (bool): Return the left and right coordinates of the
            FWHM if ``True``.

    Returns
    -------
    fwhm: float
        The calculated FWHM. It is equal to ``right - left``.
    left: float
        The position of the left of the FWHM.
    right: float
        The position of the right of the FWHM.

    """
    kernel = kernel.detach().squeeze()
    half_max = kernel.max() / 2
    indices = torch.where(kernel > half_max)[0]

    left = indices[0]
    if left > 0:
        interp = _interp(left - 1, left, kernel[left - 1], kernel[left])
        left  = interp(half_max)

    right = indices[-1]
    if right < len(kernel) - 1:
        interp = _interp(right + 1, right, kernel[right + 1], kernel[right])
        right = interp(half_max)

    fwhm = right - left
    if return_left_right:
        return fwhm, left, right
    else:
        return fwhm


def _interp(x1, x2, y1, y2):
    """1D linear interpolation."""
    a = (x1 - x2) / (y1 - y2)
    b = (x2 * y1 - x1 * y2) / (y1 - y2)
    def func(y):
        return a * y + b
    return func
