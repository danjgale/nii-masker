"""Core module that contains all functions related to extracting out time
series data.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import load_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

## CONFOUND REGRESSOR FUNCTIONS

def _compute_motion_derivs():
    """Compute derivatives from motion parameters"""
    pass


def _build_regressors(fname, regressor_names, motion_derivatives=False):
    """Create regressors for masking"""
    all_regressors = pd.read_csv(fname, sep=r'\t')
    regressors = all_regressors[regressor_names]
    # if motion_derivatives:
    #     regressors = _compute_motion_derivs(regressors)
    return regressors.values


## MASKING FUNCTIONS

def _set_masker(mask_img, **kwargs):
    """Check and see if multiple ROIs exist in atlas file"""
    n_rois = np.unique(mask_img.get_data())
    print('  {} region(s) detected from ROI file'.format(len(n_rois) - 1))

    if len(n_rois) > 2:
        masker = NiftiLabelsMasker(mask_img, **kwargs)
    elif len(n_rois) == 2:
        masker = NiftiMasker(mask_img, **kwargs)
    else:
        # only 1 value found
        raise ValueError('No ROI detected; check ROI file')
    return masker


def _mask(masker, img, regressor_names=None, roi_labels=None, as_voxels=False):
    """Extract timeseries from an image and apply post-processing"""
    timeseries = masker.fit_transform(img, confounds=regressor_names)

    if isinstance(masker, NiftiMasker):
        if as_voxels:
            labels = ['voxel{}'.format(i)
                      for i in np.arange(timeseries.shape[1])]
        else:
            timeseries = np.mean(timeseries, axis=1)
            labels = ['roi'] if roi_labels is None else roi_labels
    else:
        labels = masker.labels_ if roi_labels is None else roi_labels

    return pd.DataFrame(timeseries, columns=[str(i) for i in labels])


def _discard_initial_scans(img, n_scans, regressors=None):
    """Remove first number of scans from functional image and regressors"""
    # crop scans from functional
    arr = img.get_data()
    arr = arr[:, :, :, n_scans:]
    out_img = nib.Nifti1Image(arr, img.affine)

    if regressors is not None:
        # crop from regressors
        out_reg = regressors.iloc[n_scans:, :]
    else:
        out_reg = None

    return out_img, out_reg


def extract_data(input_files, mask_img, output_dir, labels=None,
                 regressor_files=None, regressor_names=None, as_voxels=False,
                 discard_scans=None, **masker_kwargs):
    """Extract timeseries data from input files using an roi file to demark
    the region(s) of interest(s).

    Parameters
    ----------
    input_files : list of niimg-like
        List of input NIfTI functional images
    roi_file : niimg-like
        Image that contains region mask(s). Can either be a single binary mask
        for a single region, or a numerically labeled atlas file. 0 must
        indicate background (non-region voxels).
    output_dir : str
        Save directory.
    roi_labels : str or list of str
        ROI names which are in order of ascending numeric labels in roi_file.
        Default is None
    regressor_files : list of str, optional
        Confound .csv files for each run. Default is None
    regressors : list of str, optional
        Regressor names to select from `regressor_files` headers. Default is
        None
    as_voxels : bool, optional
        Extract out individual voxel timecourses rather than mean timecourse of
        the ROI, by default False. NOTE: This is only available for binary masks,
        not for atlas images (yet)
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_img = load_img(mask_img)
    masker = _set_masker(mask_img, **masker_kwargs)

    for i, img in enumerate(input_files):

        basename = os.path.basename(img)
        print('  Extracting from {}'.format(basename))
        img = nib.load(img)

        if regressor_files is not None:
            confounds = _build_regressors(regressor_files[i], regressor_names)
        else:
            confounds = None

        if (discard_scans is not None) | (discard_scans > 0):
            img, confounds = _discard_initial_scans(img, discard_scans,
                                                    confounds)

        data = _mask(masker, img, confounds, labels, as_voxels)

        out_fname = basename.split('.')[0] + '_timeseries.tsv'
        data.to_csv(os.path.join(output_dir, out_fname), sep='\t', index=False)
