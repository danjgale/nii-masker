# nii-masker

This is a simple command-line wrapper for `nilearn`'s [Masker objects](https://nilearn.github.io/manipulating_images/masker_objects.html), which let you easily extract out region-of-interest (ROI) timeseries from functional MRI data while providing several options for applying additional post-processing (e.g., spatial smoothing, temporal filtering, confound regression, etc). This tool ultimately aims to extend many of `nilearn`'s powerful and convenient masking features to non-Python users who wish to analyze fMRI data.

# Documentation

## Installation
First, download this repository to a directory. Then, navigate to the directory, `nii-masker/`, and run `pip install .` to install `niimasker`. To check your installation, run `niimasker -h` and you should see the help information.

## Running `niimasker`
`niimasker` can be run via the command-line and can take the following arguments:

```
usage: niimasker [-h] [-i input_files [input_files ...]] [-m mask_img]
                 [--labels labels [labels ...]]
                 [--regressor_files regressor_files [regressor_files ...]]
                 [--regressor_names regressor_names [regressor_names ...]]
                 [--motion_derivs motion_derivs] [--as_voxels as_voxels]
                 [--standardize standardize] [--t_r t_r]
                 [--high_pass high_pass] [--low_pass low_pass]
                 [--detrend detrend] [--smoothing_fwhm smoothing_fwhm]
                 [--discard_scans discard_scans] [-c config]
                 output_dir

positional arguments:
  output_dir            The path to the output directory

optional arguments:
  -h, --help            show this help message and exit
  -i input_files [input_files ...], --input_files input_files [input_files ...]
                        One or more input NIfTI images. Can also be a single
                        string with a wildcard (*) to specify all files
                        matching the file pattern. If so, these files are
                        naturally sorted by file name prior to extraction.
  -m mask_img, --mask_img mask_img
                        File path of the atlas/ROI mask. Can either be a
                        single ROI mask that is binary, or an atlas with
                        numeric labels. Must be a sinlge NIfTI file in the
                        same space as the input images.
  --labels labels [labels ...]
                        Labels corresponding to the mask numbers in `mask`.
                        They must be sorted in ascending order to correctly
                        correspond to the atlas indices. The number of labels
                        provided must match the number of non-zero indices in
                        `mask`. If none are provided, numeric indices are used
  --regressor_files regressor_files [regressor_files ...]
                        One or more tabular files with regressors in each
                        column. The number of files match the number of input
                        NIfTI files provided and must be in the same order.
                        The number of rows in each file must match the number
                        of timepoints in their respective input NIfTI files.
                        Can also be a single string with a wildcard (*) to
                        specify all files matching the file pattern. If so,
                        these files are naturally sorted by file name prior to
                        extraction. Double check to make sure these are
                        correctly aligned with the input NIfTI files.
  --regressor_names regressor_names [regressor_names ...]
                        The regressor names to use for confound regression.
                        Applies to all regressor files and the names must
                        correspond to headers in each file
  --motion_derivs motion_derivs
                        Whether to include temporal derivatives of motion
                        regressors. --t_r must be specified.
  --as_voxels as_voxels
                        Whether to extract out the timeseries of each voxel
                        instead of the mean timeseries. This is only available
                        for single ROI binary masks. Default False.
  --standardize standardize
                        Whether to standardize (z-score) each timeseries.
                        Default False
  --t_r t_r             The TR of the input NIfTI files, specified in seconds.
                        Must be included if temporal filtering or motion
                        derivatives are specified.
  --high_pass high_pass
                        High pass filter cut off in Hertz. If notspecified, no
                        filtering is done.
  --low_pass low_pass   Low pass filter cut off in Hertz. If not specified, no
                        filtering is done.
  --detrend detrend     Whether to detrend the data. Default False
  --smoothing_fwhm smoothing_fwhm
                        Smoothing kernel FWHM (in mm) if spatial smoothing is
                        desired. If not specified, no smoothing is performed.
  --discard_scans discard_scans
                        Discard the first N scans of each functional NIfTI
                        image.
  -c config, --config config
                        Configuration .json file as an alternative to command-
                        line arguments. See online documentation for what keys
                        to include.
```

Most of the parameters map directly onto the Masker function arguments in `nilearn` (see the [documentation](https://nilearn.github.io/modules/reference.html#module-nilearn.input_data) and [user guide](https://nilearn.github.io/building_blocks/manual_pipeline.html#masking) for more detail). Additionally, `--discard_scans` lets you remove the first *N* scans of your data prior to extraction, `--as_voxels` lets you get individual voxel timeseries when using a single ROI, and `--labels` lets you
label your ROIs instead or just using the numerical indices.

Of course, if you want full `nilearn` flexibility, you're better off using `nilearn` and Python directly.

**Required parameters**
-  `ouput_dir`, specified by command-line only
- `input_files`, can be specified by the command-line or by a configuration file
- `mask_img`, can be specified by the command-line or by a configuration file

All other arguments are optional.

## The configuration JSON file

Instead of passing all of the parameters through the command-line, `niimasker` also provides support for a simple configuration JSON file. The only parameter that needs to be passed into the command-line is the output directory (`output_dir`). All other parameters can either be set by the configuration file or by the command-line. **Note that the configuration file overwrites any of the command-line parameters**. An empty configuration file template of all of the parameters is provided in `config_template.json`, which is shown below:

```JSON
{
  "mask_img": "",
  "labels": "",
  "regressor_files": null,
  "regressor_names": null,
  "motion_derivs": false,
  "as_voxels": false,
  "standardize": false,
  "t_r": null,
  "high_pass": null,
  "low_pass": null,
  "detrend": false,
  "smoothing_fwhm": null
}
```

An example use-case that combines both the command-line parameters and configuration file:

`niimask output/ -i img_1.nii.gz img_2.nii.gz -c config.json`

Where `config.json` is:

```JSON
{
  "mask_img": "some_atlas.nii.gz",
  "standardize": true,
  "regressor_files": [
    "confounds1.tsv",
    "confounds2.tsv"
  ],
  "regressor_names": [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "wm",
    "csf"
  ],
  "t_r": 2,
  "high_pass": 0.01,
  "smoothing_fwhm": 6
}
```
Note that you do not need to include all keys in your configuration file, just what you need.

This set up is convenient when your `output_dir` and `input_files` vary on a subject-by-subject basis, but your post-processing and atlas might stay constant across subjects and are thus stored in the project's configuration file. The configuration file therefore helps you keep track of what you did to extract out the timeseries.

# Upcoming features
- Built-in support for atlases that can be fetched directly from `nilearn`
- A comprehensive visual report (as an `html` file) in order to get a birds-eye view of the timecourses. Easily check for quality issues (e.g., amplitude spikes from motion) and how the data were generated
- Option to include event files (similar to what SPM or FSL require for first-level analyses) that labels each timepoint based on the task and conditions (only relevant for task-based fMRI).
- Full [fmriprep](https://fmriprep.readthedocs.io/en/stable/) and [BIDS](http://bids.neuroimaging.io/) support, such that confound and event files are detected automatically based on the input NIfTI. Essentially this tool could be converted to a [bids-app](http://bids-apps.neuroimaging.io/).
