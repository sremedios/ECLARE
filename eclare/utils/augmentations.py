import numpy as np
import itertools


def augment_3d_image(image):
    """
    As augmentation, we hold all combinations of horizontal and vertical flips
    as well as transposition for the in-plane of a 3D image.

    This function is written by GPT-4 Code Interpreter and edited by a human.
    """
    assert len(image.shape) == 3, "Input must be a 3D numpy array."

    augmented_images = []
    for flip_h, flip_v, do_transpose in itertools.product([False, True], repeat=3):
        # Copy the original image to avoid modifying it
        augmented_image = np.copy(image)

        # Flip horizontally if flip_h is True
        if flip_h:
            augmented_image = np.fliplr(augmented_image)

        # Flip vertically if flip_v is True
        if flip_v:
            augmented_image = np.flipud(augmented_image)

        # Transpose in the AxA plane if do_transpose is True
        if do_transpose:
            augmented_image = np.transpose(augmented_image, axes=(1, 0, 2))

        augmented_images.append(np.copy(augmented_image))

    return augmented_images
