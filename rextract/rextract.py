"""[summary]
"""

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

## CONFOUND REGRESSOR FUNCTIONS

def _compute_motion_derivs():
    """Compute derivatives from motion parameters"""
    pass


def _build_regressors(fname, regressor_names, motion_derivatives=False):
    """Create regressors for masking"""
    all_regressors = pd.read_csv(fname, sep=r'\t')
    regressors = all_regressors[regressor_names]
    if motion_derivatives:
        regressors = _compute_motion_derivs(regressors)
    return regressors


## MASKING FUNCTIONS

def _set_masker(roi_img, **kwargs):
    """Check and see if multiple ROIs exist in atlas file"""

    n_rois = np.unique(roi_img.get_data())
    print('{n_rois} regions detected from ROI file')

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
            labels = ['voxel{i}' for i in np.arange(timeseries.shape[1])]
        else:
            timeseries = np.mean(timeseries, axis=1)
            labels = 'roi' if roi_labels is None else roi_labels
    else:
        labels = masker.labels_ if roi_labels is None else roi_labels

    return pd.DataFrame(timeseries, columns=labels)


def extract_data(input_files, roi_file, roi_labels, regressor_files,
                 regressors, as_voxels=False, **masker_kwargs):
    """[summary]

    Parameters
    ----------
    input_files : [type]
        [description]
    roi_file : [type]
        [description]
    roi_labels : [type]
        [description]
    regressor_files : [type]
        [description]
    regressors : [type]
        [description]
    as_voxels : bool, optional
        [description], by default False
    """
    masker = _set_masker(roi_file, **masker_kwargs)

    for i, img in enumerate(input_files):
        regressors = _build_regressors(regressor_files[i], regressors)
        data = _mask(masker, img, regressors, roi_labels, as_voxels)






