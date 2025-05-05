# ECLARE: Efficient cross-planar learning for anisotropic resolution enhancement | [Paper](https://doi.org/10.1007/978-3-031-44689-4_12)
ECLARE is a self super-resolution method; no outside training data is required.
The only required input is the image itself and the output directory.

The journal manuscript is currently in preparation; the link will be provided soon!

NOTE: ECLARE was developed on top of refactors and improvements to [SMORE](https://gitlab.com/iacl/smore.git) and as such the SASHIMI 2023 paper points to that URL. However the code and developers are the same.

## Quick start
We recommend starting from a fresh Python 3.10 installation. If you use Anaconda or Miniconda, you can do:

```conda create -n eclare python=3.10```

Although this code may work on version of Python >3.10, it has not been tested so results may vary on such software versions. 

Run `pip install eclare`

To super-resolve an input, run:

```run-eclare --in-fpath ${INPUT_FPATH} --out-dir ${OUTPUT_DIR} --gpu-id ${GPU_ID}```

This will produce a file ending in `{YOUR_INPUT_FILENAME}_eclare.nii.gz` in `OUTPUT_DIR` and will use the GPU you specify.

We also support additional arguments:

```run-eclare --in-fpath ${INPUT_FPATH} --out-dir ${OUTPUT_DIR} [--relative-slice-thickness ${SLICE_THICKNESS}] [--relative-slice-profile-fpath ${BLUR_KERNEL_FILE}] [--gpu-id ${GPU_ID}] [--suffix ${SUFFIX}]```

where each argument in `[]` is optional and may be omitted.
- If you have a good guess of your data's slice thickness, you can specify `--relative-slice-thickness` and provide a number relative to the in-plane. So if in-plane is, say, 0.125mm and you think through-plane is 0.25mm, then you would give `--relative-slice-thickness 2`. This models the relative slice profile as a Gaussian kernel, but slice separation is still read from the header. Therefore, setting this parameter indirectly specifies slice gap as well. 
- The default GPU ID is `0`. 
- You can change the suffix to be `-SR` to produce `[...]-SR.nii.gz`, etc., based on your needs. You specify the delimiter.
- If you know the form of your slice selection profile as a `.npy` array, you can also specify it with `--relative-slice-profile-fpath`. 

## Citations
If this work is useful to you or your project, please consider citing us!

```
Remedios, S.W., Han, S., Zuo, L., Carass, A., Pham, D.L., 
Prince, J.L. and Dewey, B.E., 2023, October. Self-supervised 
super-resolution for anisotropic MR images with and without 
slice gap. In International Workshop on Simulation and 
Synthesis in Medical Imaging (pp. 118-128). Cham: Springer 
Nature Switzerland.
```

```
@inproceedings{remedios2023self,
  title={Self-supervised super-resolution for anisotropic {MR} images with and without slice gap},
  author={Remedios, Samuel W and Han, Shuo and Zuo, Lianrui and Carass, Aaron and Pham, Dzung L and Prince, Jerry L and Dewey, Blake E},
  booktitle={International Workshop on Simulation and Synthesis in Medical Imaging},
  pages={118--128},
  year={2023},
  organization={Springer}
}
```
