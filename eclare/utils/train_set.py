import numpy as np
import torch
from torch.utils.data import Dataset
from math import ceil
import random

import torch.nn.functional as F
from radifox.utils.resize.pytorch import resize

from .augmentations import augment_3d_image
from .patch_ops import get_patch_edge, get_patch_center, get_random_centers
from .pad import target_pad
from .parse_image_file import lr_axis_to_z, normalize
from .timer import timer_context
from .blur_kernel_ops import calc_extended_patch_size


class TrainSet(Dataset):
    def __init__(self, image, lr_axis, verbose=True):
        self.verbose = verbose
        self.vol = lr_axis_to_z(image, lr_axis)

        # pad the volume to the biggest in-plane square
        with timer_context(
            "Padding image out to square in-plane...", verbose=self.verbose
        ):
            target_shape = (max(self.vol.shape), max(self.vol.shape), self.vol.shape[2])
            self.vol, self.pads = target_pad(
                self.vol,
                target_shape,
            )

    def init_espreso(self, patch_size, n_patches, patch_sampling="gradient"):
        self.algorithm = "ESPRESO"
        self.patch_size = patch_size
        self.throughplane_patch_size = min(self.patch_size)
        self.n_patches = n_patches

        with timer_context(
            "Padding image out to extract patches correctly...", verbose=self.verbose
        ):
            target_shape = [
                s + p
                for s, p in zip(
                    self.vol.shape,
                    (
                        max(self.patch_size),
                        max(self.patch_size),
                        self.throughplane_patch_size,
                    ),
                )
            ]

            # apply the pad
            self.espreso_vol, self.pads = target_pad(
                self.vol,
                target_shape,
                mode="zero",
            )

        with timer_context(
            "Generating (weighted) random patch centers..", verbose=self.verbose
        ):
            self.centers = get_random_centers(
                self.espreso_vol,
                (max(self.patch_size), max(self.patch_size), min(self.patch_size)),
                self.n_patches,
                weighted=patch_sampling == "gradient",
            )

        self.espreso_vol = torch.from_numpy(self.espreso_vol)

    def espreso_getitem(self, i):
        # pull a through-plane patch
        if np.random.uniform() > 0.5:
            throughplane_patch_size = (self.patch_size[0], 1, self.patch_size[1])
        else:
            throughplane_patch_size = (1, self.patch_size[0], self.patch_size[1])

        c = [random.choice(center) for center in self.centers]

        patch_throughplane = get_patch_center(
            self.espreso_vol, c, throughplane_patch_size
        ).squeeze()

        # Data augmentation: random flip along axes
        if np.random.uniform() > 0.5:
            patch_throughplane = torch.flip(patch_throughplane, [0])
        if np.random.uniform() > 0.5:
            patch_throughplane = torch.flip(patch_throughplane, [1])

        # add channel dim and return
        return patch_throughplane.unsqueeze(0)

    def __len__(self):
        return self.n_patches

    def init_eclare(
        self,
        patch_size,
        blur_kernel,
        slice_separation,
        n_patches,
        patch_sampling="gradient",
    ):
        self.algorithm = "ECLARE"
        self.patch_size = patch_size
        self.blur_kernel = blur_kernel
        self.slice_separation = slice_separation
        self.n_patches = n_patches
        self.patch_sampling = patch_sampling

        ext_patch_size, ext_patch_crop = calc_extended_patch_size(
            self.blur_kernel, self.patch_size
        )
        self.patch_size = ext_patch_size
        self.ext_patch_crop = [slice(None, None), slice(None, None), *ext_patch_crop]
        self.n_patches = n_patches

        with timer_context(
            "Padding image out to extract patches correctly...", verbose=self.verbose
        ):
            target_shape = [
                s + p
                for s, p in zip(
                    self.vol.shape, (max(self.patch_size), max(self.patch_size), 0)
                )
            ]

            # apply the pad
            self.eclare_vol, self.pads = target_pad(
                self.vol,
                target_shape,
            )

        self.eclare_vol = torch.from_numpy(self.eclare_vol)

        with timer_context(
            "Generating (weighted) random patch centers..", verbose=self.verbose
        ):
            self.centers = get_random_centers(
                self.eclare_vol,
                (max(self.patch_size), max(self.patch_size), min(self.patch_size)),
                self.n_patches,
                weighted=patch_sampling == "gradient",
            )

    def eclare_getitem(self, i):
        # find the center
        c = [random.choice(center) for center in self.centers]
        # Last index corresponds to the slice. Let's just pull the slice out
        img_slice = self.eclare_vol[:, :, c[-1]]

        # Data augmentation on the whole slice: random flip along first axis
        if np.random.uniform() > 0.5:
            img_slice = torch.flip(img_slice, [0])
            c[0] = img_slice.shape[0] - 1 - c[0]
        # Data augmentation on the whole slice: random flip along second axis
        if np.random.uniform() > 0.5:
            img_slice = torch.flip(img_slice, [1])
            c[1] = img_slice.shape[1] - 1 - c[1]
        # Data augmentation on the whole slice: permute axes
        if np.random.uniform() > 0.5:
            img_slice = torch.permute(img_slice, (1, 0))

        # Pull the extended-size patch
        patch_hr = get_patch_center(img_slice, (c[0], c[1]), self.patch_size)
        patch_hr = patch_hr.unsqueeze(0).unsqueeze(1)

        # Blur the patch
        patch_blur = F.conv2d(patch_hr, self.blur_kernel, padding="same")

        # Crop to true patch size
        patch_hr = patch_hr[self.ext_patch_crop]
        patch_blur = patch_blur[self.ext_patch_crop]

        # Downsample patch blur
        patch_lr = resize(patch_blur, (self.slice_separation, 1), order=3)

        patch_hr = patch_hr.squeeze(0)
        patch_lr = patch_lr.squeeze(0)
        patch_blur = patch_blur.squeeze(0)

        return patch_lr, patch_hr

    def __getitem__(self, i):
        if self.algorithm == "ESPRESO":
            return self.espreso_getitem(i)
        elif self.algorithm == "ECLARE":
            return self.eclare_getitem(i)
