# Standard imports
import argparse
import time
from pathlib import Path
from math import ceil
from tqdm.auto import tqdm

# Medical imaging imports
import nibabel as nib
from radifox.utils.resize.affine import update_affine

# ML imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .models.wdsr import WDSR

# Utils folder
from .utils.timer import timer_context
from .utils.parse_image_file import (
    lr_axis_to_z,
    z_axis_to_lr_axis,
    inv_normalize,
)
from .utils.misc_utils import LossProgBar
from .utils.patch_ops import calc_dims_to_crop, calc_dims_to_pad
from .utils.rotate import rotate_vol_2d

torch.backends.cudnn.benchmark = True


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_to_vol(model, image, batch_size):
    result = []
    for st in tqdm(range(0, image.shape[0], batch_size)):
        en = st + batch_size
        batch = image[st:en]
        with torch.inference_mode():
            with torch.amp.autocast("cuda"):
                sr = model(batch).detach().cpu()
        result.append(sr)
    result = torch.cat(result, dim=0)
    return result


set_random_seed(0)
text_div = "=" * 10


def run_eclare(
    image,
    slice_separation,
    scales,
    lr_axis,
    header,
    affine,
    orig_min,
    orig_max,
    blur_kernel,
    dataset,
    device,
    batch_size,
    n_patches=None,
    fov_aware_resampling=True,
    interp_wdsr=False,
):

    eclare_st = time.time()

    learning_rate = 1e-4

    # model setup
    model = WDSR(
        n_resblocks=16,
        num_channels=256,
        scale=slice_separation,
        interp_wdsr=interp_wdsr,
        fov_aware_resampling=fov_aware_resampling,
    ).to(device)

    p = 8
    p2 = round(slice_separation * max(model.calc_out_patch_size([p, p])))
    dataset.init_eclare((p2, p), blur_kernel, slice_separation, n_patches, fov_aware_resampling)

    n_steps = int(ceil(dataset.n_patches / batch_size))

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=learning_rate,
        total_steps=n_steps + 1,
        cycle_momentum=True,
    )
    opt.step()  # necessary for the LR scheduler
    scaler = torch.amp.GradScaler("cuda")
    loss_obj = torch.nn.MSELoss().to(device)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset automatically shuffles
        pin_memory=True,
        num_workers=min(8, torch.get_num_threads()),
    )

    # ===============================
    # ============ TRAIN ============
    # ===============================
    print(f"\n{text_div} Running ECLARE {text_div}")
    train_st = time.time()

    loss_names = ["loss"]
    pbar = LossProgBar(dataset.n_patches, batch_size, loss_names)

    for t, (patches_lr, patches_hr) in enumerate(data_loader):
        patches_hr_device = patches_hr.to(device)
        patches_lr_device = patches_lr.to(device)

        with torch.amp.autocast("cuda"):
            patches_hr_hat = model(patches_lr_device)
            loss = loss_obj(patches_hr_hat, patches_hr_device)

        # gradient updates
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        # Update progress bar
        pbar.update({"loss": loss})

    pbar.close()

    # ===============================
    # ========== INFERENCE ==========
    # ===============================

    # Uses variables instantiated way earlier
    image = lr_axis_to_z(image, lr_axis)
    image = torch.from_numpy(image)

    # ===== PREDICT =====
    angles = [0, 90]

    model_preds = []

    for i, angle in enumerate(angles):
        context_str = f"Super-resolving at {angle} degrees"
        with timer_context(context_str, verbose=False):
            # Rotate in-plane. Image starts as (hr_axis, hr_axis, lr_axis)
            image_rot = rotate_vol_2d(image.to(device), angle)
            # Ensure the LR axis is s.t. (hr_axis, C, lr_axis, hr_axis)
            image_rot = image_rot.permute(0, 2, 1).unsqueeze(1)
            # Run model
            rot_result = apply_to_vol(model, image_rot, 1)
            # Return to (hr_axis, hr_axis, lr_axis)
            result = rot_result.squeeze(1).permute(0, 2, 1)
            # Rotate back and collect
            model_preds.append(rotate_vol_2d(result, -angle))

    # ===== FINALIZE =====
    final_out = torch.mean(torch.stack(model_preds), dim=0)
    final_out = final_out.detach().cpu().numpy().astype(np.float32)
    final_out = inv_normalize(final_out, orig_min, orig_max, a=0, b=1)

    # Reorient to original orientation
    final_out = z_axis_to_lr_axis(final_out, lr_axis)

    print(f"{text_div} End prediction {text_div}")

    eclare_en = time.time()

    return final_out, model.state_dict(), eclare_en - eclare_st
