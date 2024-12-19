# Standard imports
import sys
import argparse
import time
from pathlib import Path

# Medical imaging imports
import nibabel as nib
from radifox.utils.resize.affine import update_affine

# ML imports
import numpy as np

# Main function imports
from .eclare import run_eclare
from .espreso import run_espreso

# Utils folder
from .utils.train_set import TrainSet
from .utils.timer import timer_context
from .utils.parse_image_file import parse_image, normalize
from .utils.misc_utils import parse_device

text_div = "=" * 10


def main(args=None):

    main_st = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--in-fpath", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--inplane-acq-res", type=float, nargs=2, default=None)
    parser.add_argument("--n-taps-espreso", type=int, default=21)
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--n-patches", type=int, default=832000)
    parser.add_argument("--n-subsequent-patches", type=int, default=83200)
    parser.add_argument("--patch-sampling", type=str, default="gradient")
    parser.add_argument("--suffix", type=str, default="_eclare")
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--save-intermediate", action="store_true", default=False)

    args = parser.parse_args(args if args is not None else sys.argv[1:])

    device = parse_device(args.gpu_id)
    inplane_acq_res = args.inplane_acq_res

    in_fpath = Path(args.in_fpath).resolve()
    subj_id = in_fpath.name.split(".nii")[0]
    out_fname = in_fpath.name.replace(".nii", f"{args.suffix}.nii")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    espreso_psf_fpath = out_dir / f"{subj_id}_espreso_psf.npy"
    espreso_psf_plot_fpath = out_dir / f"{subj_id}_espreso_psf_plot.png"
    eclare_out_fpath = out_dir / out_fname

    with timer_context("Parsing image file...", verbose=False):
        image, slice_separation, scales, lr_axis, header, affine, orig_min, orig_max = (
            parse_image(in_fpath, normalize_image=True, inplane_acq_res=inplane_acq_res)
        )

    # First iteration defaults
    train_vol = image
    model_state = None
    n_patches = None

    for i in range(1, args.n_iters + 1):

        dataset = TrainSet(image=train_vol, lr_axis=lr_axis, verbose=False)

        if i == 1:
            # ========== ESPRESO ==========
            g, espreso_elapsed_time = run_espreso(
                slice_separation,
                args.n_taps_espreso,
                dataset,
                device,
                espreso_psf_fpath,
                espreso_psf_plot_fpath,
            )

            blur_kernel = g().detach().cpu()

        # ========== ECLARE ==========
        if i > 1:
            n_patches = args.n_subsequent_patches
        else:
            n_patches = args.n_patches

        eclare_out_fpath_iter = Path(
            str(eclare_out_fpath).replace(".nii", f"_iter{i}.nii")
        )

        eclare_vol, model_state, eclare_elapsed_time = run_eclare(
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
            eclare_out_fpath_iter,
            n_patches=n_patches,
            model_state=model_state,
            save_intermediate=args.save_intermediate,
        )

        train_vol, *_ = normalize(eclare_vol)

        # Print final timings and wrap-up.
        print(
            (
                f"\nFinished iteration {i} / {args.n_iters}\n"
                f"\tESPRESO elapsed time: {espreso_elapsed_time:.4f}s\n"
                f"\tECLARE elapsed time: {eclare_elapsed_time:.4f}s\n"
            )
        )

    print("Saving image...")
    # Update affine matrix
    scales[lr_axis] = 1 / slice_separation
    new_affine = update_affine(affine, scales)

    # Write nifti
    out_obj = nib.Nifti1Image(eclare_vol, affine=new_affine, header=header)
    nib.save(out_obj, eclare_out_fpath)

    main_en = time.time()

    # Print final timings and wrap-up.
    print(
        (
            "\n\nDONE\n"
            f"Total elapsed time: {main_en - main_st:4f}s\n"
            f"\tWritten to: {eclare_out_fpath}\n"
            f"{text_div} END PREDICTION {text_div}"
        )
    )
