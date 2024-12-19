# ECLARE: Efficient cross-planar learning for anisotropic resolution enhancement | [Paper](https://doi.org/10.1007/978-3-031-44689-4_12)
ECLARE is a self super-resolution method; no outside training data is required.
The only required input is the image itself and the output directory.

The journal manuscript is currently in preparation; the link will be provided soon!

NOTE: ECLARE was developed on top of refactors and improvements to [SMORE](https://gitlab.com/iacl/smore.git) and as such the SASHIMI 2023 paper points to that URL. However the code and developers are the same.

## Installation from Source

### Install using `pip`
We recommend starting from a fresh Python 3.10 installation. If you use Anaconda or Miniconda, you can do:

```conda create -n eclare python=3.10```

Although this code may work on version of Python >3.10, it has not been tested so results may vary on such software versions. 

Clone the [repository](https://github.com/sremedios/ECLARE) and navigate to the root project directory.
Run:

 ```pip install .```

Package requirements are automatically handled. To see a list of requirements, see `setup.py` L50-59.
This installs the `eclare` package and creates two CLI aliases `eclare-train` and `eclare-test`.
 
### Basic Usage

There are two basic ways to run this code: from source after running `pip install .` or from the Singularity container.

```run-eclare --in-fpath ${INPUT_FPATH} --out-dir ${OUTPUT_DIR} [--slice-thickness ${SLICE_THICKNESS}] [--blur-kernel-file ${BLUR_KERNEL_FILE}] [--gpu-id ${GPU_ID}] [--suffix ${SUFFIX}]```

where each argument in `[]` is optional and may be omitted. Remember with Singularity that you need to bind the right drives with `-B ${DRIVE_ROOT}` where `${DRIVE_ROOT}` is where your data sits. The default GPU ID is `0`. The container will automatically handle whether the slice thickness or blur kernel file is present, or neither case. The `--suffix` argument allows you to specify the suffix of the output filename before the file extension. Default is `_eclare`.

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
