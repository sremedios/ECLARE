import numpy as np


def get_pads(target_dim, d):
    if target_dim <= d:
        return 0, 0
    p = (target_dim - d) // 2
    if (p * 2 + d) % 2 != 0:
        return p, p + 1
    return p, p


def target_pad(img, target_dims, mode="zero"):
    pads = tuple(get_pads(t, d) for t, d in zip(target_dims, img.shape))

    if mode == "zero":
        return np.pad(img, pads, mode="constant", constant_values=0), pads
    return np.pad(img, pads, mode=mode), pads


def format_pads(pads):
    """Turn pad amounts into appropriate slices and handle 0 pads as None slices"""
    st = pads[0] if pads[0] != 0 else None
    en = -pads[1] if pads[1] != 0 else None
    return slice(st, en)


def crop(img, pads):
    crops = tuple(map(format_pads, pads))
    return img[crops]
