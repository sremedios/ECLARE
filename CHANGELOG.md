# Changelog
All notable changes to `smore` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.0] - 2024-04-01
### Changed
- Reflect padding is now edge-padding througout
- No longer uses OneCycle learning rate scheduling
- Trains for a fixed number of iterations (longer than 4.0)
- Uses wavelet loss alongside L1
- Contains support for iSMORE (but not recommended atm)
- Mathematically correct handling of padding at inference time
### Added
- Includes custom implementation of ESPRESO for robustness, speed, and simplicity


## [4.0.0] - 2023-04-19
SMORE v4 implements a refactor of S-SMORE with some more changes.
- Single-network like S-SMORE (only SR network, no more AA network)
- Backbone is now WDSR instead of EDSR
- Like S-SMORE, the input is first interpolated to the nearest integer factor grid size, then "pixel shuffle" is used to do the remaining integer upsampling
- "Patch size" specifies the low-res patch size now and is scaled up to pull high-res patches
- No more reliance on pre-trained weights; they're not even provided
- Support for a physics-based slice selection profile `rf-pulse-slr` as the default
- Supports rational slice separation

## [3.2.3] - 2022-10-28
### Changed
- Bugfix to correctly calculate the FWHM of the blur kernel to account for the underlying spatial resolution per pixel

## [3.2.2] - 2022-06-10
### Added
- Added CHANGELOG.md to keep track of changes
### Changed
- Removed old versioning files
- Update weights file extension to use the standard ".pt" extension
- Split example scripts into container and source scripts
- Updated README with more detailed install instructions (git-lfs)

## [3.2.1] - 2022-05-25
### Added
- Added automatic versioning using `miniver`
### Changed
- Changed CI to build on all tags, `main` branch and all `support/*` branches
- Changed contribution policy to follow GithubFlow

## [3.2.0] - 2022-05-24
This is a bugfix release for SMOREv3 that corrects the affine matrix after running SMORE.
This **does not affect the numerical values of the image array**, but rather how the image is compared to other images in some visualization software.
The issue stems from a require update to the image origin to take into account the different resolution of the voxels.
### Fixed
- Affine matrix is now updated with new origin based on the resolution change.
### Changed
- General cleanup of the repository
- Added CI to build Docker images of tags/main branches and included weights in LFS repo

## [3.1.2] - 2021-09-21
### Fixed
- Corrected header writing for images which are not LR on the last axis
- Fixed cropping at test time

## [3.1.1] - 2021-06-15
This is a bugfix release for SMOREv3 to correct an issue where for some volume shape/downsample ratio combinations, the output shape after aliasing would not be the same as the original shape.
This is due to a rounding operation during downsampling.
This fix sets the output shape of the upsampling function to add in these "missing" voxels.
### Fixed
- Set output shape of the upsampling function to be the same as the original shape of the image during aliasing step.

## [3.1.0] - 2021-06-08
### Added
- Able to specify slice spacing, slice thickness, and blur kernel for volumes which are already isotropic
- Able to supply a custom blur kernel
### Changed
- Training patches are now sampled by gradient weights
- Code refactor for easier maintenance

## [3.0.1] - 2021-06-03
This is a hot fix release for SMOREv3 based on some runtime debugging.
### Change
- Updated README.md to reflect new options and include relevant citations
- Updated `scripts/run_smore.sh`
- Changed bash and train/test script arguments from `_` delimiters to `-` delimiters
- Changed the argument name `n_items` to `n_patches` for clarity
- Further minor code cleanups

## [3.0.0] - 2021-05-27
SMORE v3 implements a complete overhaul of the code to use Pytorch instead of Tensorflow v1.
This was done to improve development time and code readability.
**WARNING:** This version is the initial release and may be unstable. Use the `v3.0.1` for a more stable release.

## [2.1.1] - 2021-05-12
In the 2.x.x series, we include support for iterative SMORE (iSMORE).
This allows the user to specify some number of iterations to continue training.
Recall that in SMORE, we train on in-plane patches and test on through-plane slices which have been interpolated.
The in-plane patches represent "thick" data, but the through-plane slices represent "thin" data.
To address this issue, iSMORE fine-tunes itself in subsequent iterations by using the super-resolved volume to generate new training data.
The super-resolved volume's in-plane patches are now "thinner" than the original, so these should serve as better training data (ie: closer to the testing distribution).
After all the iterations have completed, the model is applied to the original volume.
In 2.1.x, we include support for the user to provide a slice selection profile kernel.
This is a kernel which represents the slice selection profile used in 2D MR acquisitions and can compensate for slice gap and slice overlap.
This is commonly estimated from the slice profile estimation software ESPRESO.
### Fixed
- Training from scratch properly fine-tunes the model on iterations > 1
- We no longer user scipy.ndimage.zoom for interpolation due to incorrect assumptions.
We now explicitly set the step and interpolate at query coordinates
- Fourier burst accumulation re-implemented to match the paper's equations.

## [1.0.0] - 2020-07-01
Initial version of SMORE before iterative SMORE was released.

