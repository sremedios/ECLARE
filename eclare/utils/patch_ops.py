import numpy as np
from scipy.ndimage import gaussian_filter
from math import ceil, floor


def get_ilb(m, s, a, b):
    return round(m * s) / a - m - 1 / (2 * a)


def get_plb(m, s, i, a, b):
    return a * i / b - 1 / (2 * b) + a * m / b - round(s * m) / b


def calc_dims_to_pad(m, s):
    epsilon = 1e-8

    if s <= 2:
        return 0

    a = floor(s)
    b = s - a

    if b == 0:
        return 0

    # First case: c_1 <= pi <= c2
    i_lb = get_ilb(m, s, a, b)
    i = ceil(i_lb)
    p_lb = get_plb(m, s, i, a, b)
    p = ceil(max(p_lb, 0))

    if projected_size(m, p, s) == ideal_size(m, s):
        return p

    # Second case: c_1 < pi <= c2
    if abs(round(p_lb) - p_lb) < epsilon:
        p = ceil(max(p_lb + epsilon, 0))

    if projected_size(m, p, s) == ideal_size(m, s):
        return p

    # Third case: c_1 <= pi < c2
    if int(i_lb) == i_lb:
        i = ceil(i_lb + epsilon)
    p_lb = a * i / b - 1 / (2 * b) + a * m / b - round(s * m) / b
    p = ceil(max(p_lb, 0))

    if projected_size(m, p, s) == ideal_size(m, s):
        return p

    # Fourth case: c_1 < pi < c2
    p_lb = a * i / b - 1 / (2 * b) + a * m / b - round(s * m) / b
    p = ceil(max(p_lb, 0))
    if abs(round(p_lb) - p_lb) < epsilon:
        p = ceil(max(p_lb + epsilon, 0))

    if projected_size(m, p, s) == ideal_size(m, s):
        return p


def projected_size(n_slices, p, scale):
    """
    The projected number of slices after initially padding `n_slices`
    by `p`. We would like to choose `p` to match the results of `ideal_slice()`.
    """
    scale_tilde = scale / floor(scale)
    return round((n_slices + p) * scale_tilde) * floor(scale) - round(p * scale)


def ideal_size(n_slices, scale):
    """
    The correct number of slices according to `resize`, which uses
    a `round` operation to get an integer number of slices.
    """
    return round(n_slices * scale)


def find_integer_p(n_slices, s):
    """
    The goal here is to, at test time, pad out the number of slices, then
    run the RCAN model, then crop off all the extra slices we got from the
    initial padding. This function finds the padding which achieves this.
    """
    p = 0  # Start testing from p = 0
    max_iter = 1000  # Maximum number of iterations to prevent infinite loop
    iter_count = 0  # Counter for the number of iterations

    while (
        projected_size(n_slices, p, s) != ideal_size(n_slices, s)
        and iter_count < max_iter
    ):
        p += 1
        iter_count += 1

    # If solution is found within max_iter iterations, return p
    if projected_size(n_slices, p, s) == ideal_size(n_slices, s):
        return p
    # If no solution is found within max_iter iterations, it is unachievable
    # and we just don't do any padding.
    return 0


def calc_dims_to_crop(x, scale):
    return round(x * scale)


def get_patch_edge(sl, patch_size):
    """
    sl: np.array, the slice to draw a patch from
    patch_size: tuple of ints, the patch size in 2D.
    """

    # Choose a random starting position
    sts = [np.random.randint(0, b - p) for b, p in zip(sl.shape, patch_size)]
    ens = [st + p for st, p in zip(sts, patch_size)]
    idx = tuple(slice(st, en) for st, en in zip(sts, ens))

    return sl[idx]


def correct_center_idx(c, edge_minor, edge_major):
    """
    Sometimes, `c` might be too close to the edge. We correct it
    here to lie within the image.
    """
    return min(max(c, edge_minor), edge_major)


def get_patch_center(vol, patch_center, patch_size, return_idx=False):
    """
    vol: np.array, the image volume
    patch_center: tuple of ints, center position of the patch
    patch_size: tuple of ints, the patch size in 3D
    """

    sts = [
        correct_center_idx(c, p // 2, d - p // 2) - p // 2 if p != 1 else c
        for c, p, d in zip(patch_center, patch_size, vol.shape)
    ]
    ens = [st + p for st, p in zip(sts, patch_size)]
    idx = tuple(slice(st, en) for st, en in zip(sts, ens))

    if return_idx:
        return idx

    return vol[idx]


def get_random_centers(vol, patch_size, n_centers, weighted=True):
    if weighted:
        smooth = gaussian_filter(vol, 2.0)
        grads = np.gradient(smooth)
        grad_mag = np.sum([np.sqrt(np.abs(grad)) for grad in grads], axis=0)

        # Set probability to zero at edges
        grad_mag[: patch_size[0] // 2] = 0
        grad_mag[-patch_size[0] // 2 :] = 0

        grad_mag[:, : patch_size[0] // 2] = 0
        grad_mag[:, -patch_size[0] // 2 :] = 0

        # Short-circuit if gradient magnitude is all zero; this
        # happens when the patch size is too big for the dimensions
        # of the image volume
        if grad_mag.sum() == 0:
            print(
                (
                    "Unable to calculate image gradient due to patch"
                    "size and image extents.\n"
                    "Defaulting to uniform patch sampling."
                )
            )
            grad_probs = [None for _ in vol.shape]

        else:
            # Normalize gradient magnitude to create probabilities
            grad_probs = [
                np.array(
                    [
                        np.sum(grad_mag.take(i, axis=axis))
                        for i in range(grad_mag.shape[axis])
                    ]
                )
                for axis in range(grad_mag.ndim)
            ]
            grad_probs = [g / g.sum() for g in grad_probs]

    else:
        grad_probs = [None for _ in vol.shape]

    # Generate random patch centers for each dimension
    centers = [
        np.random.choice(
            np.arange(0, img_dim),
            size=n_centers,
            p=grad_probs[axis],
        )
        for axis, img_dim in enumerate(vol.shape)
    ]
    return centers
