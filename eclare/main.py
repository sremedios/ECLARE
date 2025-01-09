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
from .utils.parse_image_file import parse_image
from .utils.misc_utils import parse_device

text_div = "=" * 10


def main(args=None):

    main_st = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--in-fpath", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--inplane-acq-res", type=float, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-patches", type=int, default=1000000)
    parser.add_argument("--patch-sampling", type=str, default="gradient")
    parser.add_argument("--suffix", type=str, default="_eclare")
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true", default=False)

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

    dataset = TrainSet(image=image, lr_axis=lr_axis, verbose=False)

    # ========== ESPRESO ==========
    g, espreso_elapsed_time = run_espreso(
        slice_separation,
        dataset,
        device,
        espreso_psf_fpath,
        espreso_psf_plot_fpath,
    )

    blur_kernel = g().detach().cpu()

    # ========== ECLARE ==========
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
        batch_size=args.batch_size,
        n_patches=args.n_patches,
    )

    # Print final timings and wrap-up.
    print(
        (
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
