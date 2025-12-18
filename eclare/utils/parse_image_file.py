import numpy as np
import nibabel as nib
from radifox.utils.degrade.degrade import fwhm_units_to_voxel_space, fwhm_needed
from radifox.utils.degrade.inplane_res import downsample_k_space


def normalize(x, a=-1, b=1):
    orig_min = x.min()
    orig_max = x.max()

    numer = (x - orig_min) * (b - a)
    denom = orig_max - orig_min

    return a + numer / denom, orig_min, orig_max


def inv_normalize(x, orig_min, orig_max, a=0.0, b=1.0, out_dtype=None):
    # x is expected to be float array in [a,b]
    # In-place math to avoid out of mem errors:
    x = np.asarray(x)

    # x = (x - a) / (b - a)
    x -= a
    x /= (b - a)

    # x = x * (orig_max - orig_min) + orig_min
    x *= (orig_max - orig_min)
    x += orig_min

    if out_dtype is not None:
        x = x.astype(out_dtype, copy=False)

    return x


def parse_image(image_file, normalize_image=False, inplane_acq_res=None):
    """
    Open the image volume file, and return pertinent information:
    - The image array as a numpy array
    - The "scale" of the anisotropy (this is the slice separation)
    - The LR axis
    - The header of the image file
    - The affine matrix of the image file
    """
    obj = nib.load(image_file)
    voxel_size = tuple(float(v) for v in obj.header.get_zooms())
    image = obj.get_fdata(dtype=np.float32)
    scales = [1, 1, 1]

    # x, y, and z are the spatial physical measurement sizes
    lr_axis = np.argmax(voxel_size)
    z = voxel_size[lr_axis]
    xy = list(voxel_size)
    xy.remove(z)
    xyz = (xy[0], xy[1], z)
    x, y, z = xyz

    # Exit if the provided image is isotropic through-plane
    assert (
        x != z and y != z
    ), f'Worst resolution found {z} matches one of the better resolutions {x} or {y}; image is "isotropic" and cannot be run through ECLARE.'

    # Correct the in-plane resolution by dropping to the acquired resolution
    if inplane_acq_res is not None:
        print(f"\tReducing in-plane resolutions:")
        print(f"\t({x:.2f}, {y:.2f}) -> ({inplane_acq_res[0]:.2f}, {inplane_acq_res[1]:.2f})")
        scales = [inplane_acq_res[0] / x, inplane_acq_res[1] / y, 1]
        downsample_shape = [int(np.round(d / s)) for d, s in zip(image.shape, scales)]
        image = downsample_k_space(image, target_shape=downsample_shape).astype(np.float32)
        x, y = inplane_acq_res

    slice_separation = round(float(z / min(x, y)), 3)

    if normalize_image:
        image, orig_min, orig_max = normalize(image, 0, 1)
    else:
        orig_min = None
        orig_max = None

    return (
        image,
        slice_separation,
        scales,
        lr_axis,
        obj.header,
        obj.affine,
        orig_min,
        orig_max,
    )


def lr_axis_to_z(img, lr_axis):
    """
    Orient the image volume such that the low-resolution axis
    is in the "z" axis.
    """
    if lr_axis == 0:
        return img.transpose(1, 2, 0)
    elif lr_axis == 1:
        return img.transpose(2, 0, 1)
    elif lr_axis == 2:
        return img


def z_axis_to_lr_axis(img, lr_axis):
    """
    Orient the image volume such that the "z" axis
    is back to the original low-resolution axis
    """
    if lr_axis == 0:
        return img.transpose(2, 0, 1)
    elif lr_axis == 1:
        return img.transpose(1, 2, 0)
    elif lr_axis == 2:
        return img
