# Standard imports
import argparse
import time
from pathlib import Path
from math import ceil
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Medical imaging imports
from radifox.utils.resize.pytorch import resize as resize_pt

# ML imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .models.espreso_g import G
from .models.espreso_d import D
from .models.losses import *


# Utils folder
from .utils.train_set import TrainSet
from .utils.fwhm import calc_fwhm


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed(0)
matplotlib.use("Agg")  # non-interactive plotting
text_div = "=" * 10


def run_espreso(
    slice_separation,
    dataset,
    device,
    espreso_psf_fpath,
    espreso_psf_plot_fpath,
):

    espreso_st = time.time()
    print(f"{text_div} Running ESPRESO {text_div}")

    lr_patch_size = [48, 48]
    batch_size = 128
    lr = 1e-4

    # ESPRESO starts with P x 16 patches as its goal. The "generator" produces a blur kernel
    # which convolves along the first axis without padding, and then downsamples along that
    # axis. Therefore P must be 16 + 2\floor{L/2} x k, where L is the kernel length and k is
    # the slice separation.
    P = 16 * slice_separation + 2 * 10
    patch_size = (int(round(P)), 16)

    # warmup data loader
    n_patches = batch_size * 100
    n_steps = int(ceil(n_patches / batch_size))

    dataset.init_espreso(patch_size, n_patches)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset automatically shuffles
        pin_memory=False,
        num_workers=min(8, torch.get_num_threads()),
    )

    g = G().to(device)
    opt_g = torch.optim.AdamW(g.parameters(), lr=lr)
    scaler_g = torch.amp.GradScaler("cuda")

    # Warm up to RF-Pulse-SLR of the slice separation
    ref_psf = torch.zeros(1, 1, 21, 1).to(device)
    ref_psf[:, :, 21 // 2, :] = 1

    loss_fn = torch.nn.MSELoss().to(device)

    for x_batch in tqdm(data_loader):

        x = x_batch.to(device)

        opt_g.zero_grad()

        with torch.amp.autocast("cuda"):
            est_psf = g()

            est = F.conv2d(x, est_psf)
            ref = F.conv2d(x, ref_psf)

            loss = loss_fn(est, ref)

        scaler_g.scale(loss).backward()

        scaler_g.step(opt_g)
        scaler_g.update()

    est_psf_warmup = g().detach().cpu().numpy().squeeze()

    ### Adversarial training

    d_patch_size = (10, 16)

    adv_loss_fn = GANLoss().to(device)
    peak_loss_fn = PeakLoss().to(device)
    bound_loss_fn = BoundaryLoss(kernel_length=21).to(device)

    # now adversarial loss.
    # "real" means that the horizontal direction is from the through-plane axis
    # "fake" means that the horizontal direction is NOT from the through-plane axis

    # adversarial training uses more training patches
    dataset.n_patches = 128 * 2 * 2000
    n_steps = int(ceil(dataset.n_patches / 128 * 2))

    # re-initialize the data_laoder with new batch size
    data_loader = DataLoader(
        dataset,
        batch_size=int(data_loader.batch_size * 2),
        shuffle=False,  # Dataset automatically shuffles
        pin_memory=True,
        num_workers=8,
    )

    # Initialize discriminator
    d = D().to(device)
    opt_d = torch.optim.AdamW(d.parameters(), lr=lr)

    # Set up LR schedulers
    sch_g = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt_g,
        max_lr=lr,
        total_steps=n_steps + 1,
        cycle_momentum=True,
    )
    sch_d = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt_d,
        max_lr=lr,
        total_steps=n_steps + 1,
        cycle_momentum=True,
    )

    # Step optimizers for schedulers
    opt_g.step()
    opt_d.step()

    # set up scalers
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")

    for x_batch in tqdm(data_loader):

        x = x_batch.to(device)

        # ===== Train generator ===== #
        # freeze discriminator
        for p in d.parameters():
            p.requires_grad = False

        opt_g.zero_grad()

        with torch.amp.autocast("cuda"):

            est_psf = g()

            y = F.conv2d(x, est_psf)
            y = resize_pt(y, (slice_separation, 1), order=3)

            real_horiz = y[: batch_size // 2].permute(0, 1, 3, 2)
            fake_horiz = y[batch_size // 2 :]

            N = real_horiz.shape[0]
            g_loss = adv_loss_fn(d(fake_horiz), is_real=True)
            p_loss = peak_loss_fn(est_psf)
            b_loss = bound_loss_fn(est_psf)

            loss = g_loss + 0.5 * p_loss + 10 * b_loss

        scaler_g.scale(loss).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        sch_g.step()

        # ===== Train discriminator ===== #
        # unfreeze the discriminator
        for p in d.parameters():
            p.requires_grad = True

        opt_d.zero_grad()

        with torch.amp.autocast("cuda"):
            # Measure discriminator's ability to classify real from generated samples
            if np.random.rand() < 0.5:
                d_loss = adv_loss_fn(d(real_horiz.detach()), is_real=True)
            else:
                d_loss = adv_loss_fn(d(fake_horiz.detach()), is_real=False)

        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        sch_d.step()

    espreso_psf = g().detach().cpu().numpy().squeeze()

    # save PSF to disk
    with open(espreso_psf_fpath, "wb") as fpointer:
        np.save(fpointer, espreso_psf)

    # save plot of PSF to disk
    fwhm = calc_fwhm(espreso_psf)[0]
    plt.plot(espreso_psf, color="blue")

    title = f"Calculated FWHM: {fwhm:.2f}"
    plt.title(title)
    plt.savefig(espreso_psf_plot_fpath)

    espreso_en = time.time()

    return g, espreso_en - espreso_st
