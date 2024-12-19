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
from .utils.train_set import TrainSet
from .utils.timer import timer_context
from .utils.parse_image_file import (
    lr_axis_to_z,
    z_axis_to_lr_axis,
    inv_normalize,
)
from .utils.misc_utils import LossProgBar
from .utils.patch_ops import calc_dims_to_crop, calc_dims_to_pad
from .utils.rotate import rotate_vol_2d


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
            with torch.cuda.amp.autocast():
                sr, *_ = model(batch)
                sr = sr.detach().cpu()
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
    eclare_out_fpath,
    n_patches=None,
    model_state=None,
    save_intermediate=True,
):

    eclare_st = time.time()

    learning_rate = 1e-3

    # model setup
    model = WDSR(
        n_resblocks=16,
        num_channels=32,
        scale=slice_separation,
    ).to(device)

    if model_state is not None:
        model.load_state_dict(model_state)

    K = 1
    top_model_states = []

    # Change params for ECLARE loading
    batch_size = 32

    # Find the patch size that works for training for this slice separation.
    # We start at the network's receptive field, 38, and find the next valid number
    p = 38
    p = calc_dims_to_pad(p, slice_separation) + p

    dataset.init_eclare(
        model.calc_out_patch_size([p, p]), blur_kernel, slice_separation, n_patches
    )

    n_steps = int(ceil(dataset.n_patches / batch_size))

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=learning_rate,
        total_steps=n_steps + 1,
        cycle_momentum=True,
    )
    opt.step()  # necessary for the LR scheduler

    scaler = torch.cuda.amp.GradScaler()
    loss_obj = torch.nn.L1Loss().to(device)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset automatically shuffles
        pin_memory=True,
        num_workers=min(8, torch.get_num_threads()),
    )

    # Train
    print(f"\n{text_div} Running ECLARE {text_div}")
    train_st = time.time()

    loss_names = ["loss"]
    pbar = LossProgBar(dataset.n_patches, batch_size, loss_names)

    for t, (patches_lr, _, patches_hr) in enumerate(data_loader):
        patches_hr_device = patches_hr.to(device)
        patches_lr_device = patches_lr.to(device)

        with torch.cuda.amp.autocast():
            patches_hr_hat, _, _ = model(patches_lr_device)
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

    # pad the voxel dims out so we achieve the correct final resolution
    ps = [calc_dims_to_pad(dim, slice_separation) for dim in image.shape]
    # Only crop off the through-plane direction based on slice separation scaling
    cs = [ps[0], ps[1], calc_dims_to_crop(ps[2], slice_separation)]
    cs = tuple([slice(None, -c) if c != 0 else slice(None, None) for c in cs])

    image = np.pad(image, ((0, ps[0]), (0, ps[1]), (0, ps[2])))
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

    # Re-crop to target shape
    final_out = final_out[cs]
    # Reorient to original orientation
    final_out = z_axis_to_lr_axis(final_out, lr_axis)

    print(f"{text_div} End prediction {text_div}")

    if save_intermediate:
        print("Saving image...")
        # Update affine matrix
        scales = scales[:-1]
        scales.insert(lr_axis, 1 / slice_separation)
        new_affine = update_affine(affine, scales)

        # Write nifti
        out_obj = nib.Nifti1Image(final_out, affine=new_affine, header=header)
        nib.save(out_obj, eclare_out_fpath)
        print("\tIntermediate result written to: {}\n".format(str(eclare_out_fpath)))

        # Also save intermediate non-avgd results
        for i, model_pred in enumerate(model_preds):
            model_pred = model_pred.detach().cpu().numpy().astype(np.float32)
            model_pred = inv_normalize(model_pred, orig_min, orig_max, a=0, b=1)

            # Re-crop to target shape
            model_pred = model_pred[cs]
            # Reorient to original orientation
            model_pred = z_axis_to_lr_axis(model_pred, lr_axis)
            # Write nifti
            out_obj = nib.Nifti1Image(model_pred, affine=new_affine, header=header)
            nib.save(out_obj, str(eclare_out_fpath).replace(".nii", f"angle{i}.nii"))

    eclare_en = time.time()

    return final_out, model.state_dict(), eclare_en - eclare_st
