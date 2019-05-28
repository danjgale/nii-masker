# nii-masker

This is a simple command-line wrapper for `nilearn`'s [Masker object](https://nilearn.github.io/manipulating_images/masker_objects.html), which lets you easily extract out region-of-interest (ROI) timeseries from functional MRI data while giving you several options to apply post-processing to your MRI data (e.g., spatial smoothing, temporal filtering, confound regression, etc). This tool ultimately aims to extend `nilearn`'s powerful and convenient masking features to non-Python users who wish to analyze fMRI data.

## Installation

First, download this repository to a directory. Then, run `pip install .` to install `niimasker`.

## Running `niimasker`

In order to run `niimasker`, you will need to specify an input directory, and output directory, and a configuration file (see `config_template.json`). This can be run into the command-line as so:

`niimasker /path/to/input /path/to/output config.json`

## Configuring `niimasker`

Coming soon.

## Upcoming features

- Expanded command-line arguments
- A comprehensive visual report (as an `html` file) in order to get a birds-eye view of the timecourses. Easily check for quality issues (e.g., amplitude spikes from motion) and how the data were generated
- Option to include event files (similar to what SPM or FSL require for first-level analyses) that labels each timepoint based on the task and conditions (only relevant for task-based fMRI).
- Full [fmriprep](https://fmriprep.readthedocs.io/en/stable/) and [BIDS](http://bids.neuroimaging.io/) support, such that confound and event files are detected automatically based on the input NIfTI.
