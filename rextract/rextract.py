"""Core module that contains all functions related to extracting out time
series data.
"""

import os
import numpy as np
import pandas as pd
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

def _set_masker(roi_img, **kwargs):
    """Check and see if multiple ROIs exist in atlas file"""

    n_rois = np.unique(roi_img.get_data())
    print('{} regions detected from ROI file'.format(n_rois))

    if len(n_rois) > 2:
        masker = NiftiLabelsMasker(roi_img, **kwargs)
    elif len(n_rois) == 2:
        masker = NiftiMasker(roi_img, **kwargs)
    else:
        # only 1 value found
        raise ValueError('No ROI detected; check ROI file')
    return masker


def _mask(masker, img, regressors=None, roi_labels=None, as_voxels=False):
    """Extract timeseries from an image and apply post-processing"""

    timeseries = masker.fit_transform(img, confounds=regressors)

    if isinstance(masker, NiftiMasker):
        if as_voxels:
            labels = ['voxel{}'.format(i)
                      for i in np.arange(timeseries.shape[1])]
        else:
            timeseries = np.mean(timeseries, axis=1)
            labels = 'roi' if roi_labels is None else roi_labels
    else:
        labels = masker.labels_ if roi_labels is None else roi_labels

    return pd.DataFrame(timeseries, columns=labels)


def extract_data(input_files, roi_file, output_dir, roi_labels=None,
                 regressor_files=None, regressors=None, as_voxels=False,
                 **masker_kwargs):
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
    masker = _set_masker(roi_file, **masker_kwargs)

    for i, img in enumerate(input_files):
        regressors = _build_regressors(regressor_files[i], regressors)
        data = _mask(masker, img, regressors, roi_labels, as_voxels)
        out_fname = os.path.basename(input_files).split('.')[0] + '_timeseries.tsv'
        data.to_csv(os.path.join(output_dir, out_fname), sep='\t', index=False)






