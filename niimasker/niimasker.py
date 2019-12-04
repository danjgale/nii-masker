"""Core module that contains all functions related to extracting out time
series data.
"""

import os
from itertools import repeat
import multiprocessing
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import load_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

from niimasker.report import generate_report


class FunctionalImage(object):
    def __init__(self, fname):

        self.fname = fname
        img = nib.load(self.fname)
        self.img = img

        self.regressors = None
        self.regressor_file = None

    def set_regressors(self, regressor_fname, regressor_labels=None):
        """Create regressors for masking"""
        self.regressor_file = regressor_fname
        all_regressors = pd.read_csv(regressor_fname, sep=r'\t',
                                     engine='python')
        if regressor_labels is not None:
            self.regressors = all_regressors[regressor_labels]
        else:
            self.regressors = all_regressors


    def discard_scans(self, n_scans):
        # crop scans from image
        arr = self.img.get_data()
        arr = arr[:, :, :, n_scans:]
        self.img = nib.Nifti1Image(arr, self.img.affine)

        if self.regressors is not None:
            # crop from regressors
            self.regressors = self.regressors.iloc[n_scans:, :]


    def extract(self, masker, as_voxels=False, roi_labels=None):
        print('  Extracting from {}'.format(os.path.basename(self.fname)))

        if self.regressors is None:
            timeseries = masker.fit_transform(self.img)
        else:
            timeseries = masker.fit_transform(self.img,
                                              confounds=self.regressors.values)

        # determine column names for timeseries
        if isinstance(masker, NiftiMasker):
            labels = ['voxel {}'.format(int(i))
                      for i in np.arange(timeseries.shape[1])]
            self.mask_img = masker.mask_img_
        
        else:
            # multiple regions from an atlas were extracted
            if roi_labels is None:
                labels = ['roi {}'.format(int(i)) for i in masker.labels_]
            else:
                labels = roi_labels

            self.mask_img = masker.labels_img

        self.masker = masker
        self.data = pd.DataFrame(timeseries, columns=[str(i) for i in labels])
        self.voxelwise = as_voxels


## MASKING FUNCTIONS

def _set_masker(mask_img, as_voxels=False, **kwargs):
    """Check and see if multiple ROIs exist in atlas file"""
    n_rois = np.unique(mask_img.get_data())
    print('  {} region(s) detected from {}'.format(len(n_rois) - 1,
                                                   mask_img.get_filename()))

    if len(n_rois) > 2:
        
        if as_voxels:
            raise ValueError('`as_voxels` must be set to False (Default) if '
                             'using an mask image with > 1 region. ')
        else:
            # mean timeseries extracted from regions
            masker = NiftiLabelsMasker(mask_img, **kwargs)
    elif len(n_rois) == 2:
        # single binary ROI mask 
        if as_voxels:
            masker = NiftiMasker(mask_img, **kwargs)
        else:
            # more computationally efficient if only wanting the mean of ROI
            masker = NiftiLabelsMasker(mask_img, **kwargs)
    else:
        # only 1 value found
        raise ValueError('No ROI detected; check ROI file')
    return masker


def _mask_and_save(masker, img_name, output_dir, regressor_file=None,
                   regressor_names=None, as_voxels=False,
                   labels=None, discard_scans=None):
    """Runs the full masking process and saves output for a single image;
    the main function used by `make_timeseries`"""
    # basename = os.path.basename(img_name)
    # print('  Extracting from {}'.format(basename))
    # img = nib.load(img_name)

    img = FunctionalImage(img_name)

    if regressor_file is not None:
        img.set_regressors(regressor_file, regressor_names)

    if discard_scans is not None:
        if discard_scans > 0:
            img.discard_scans(discard_scans)

    img.extract(masker, as_voxels=as_voxels, roi_labels=labels)

    # export data and report
    out_fname = os.path.basename(img.fname).split('.')[0] + '_timeseries.tsv'
    img.data.to_csv(os.path.join(output_dir, out_fname), sep='\t', index=False,
                    float_format='%.8f')
    generate_report(img, output_dir)



def make_timeseries(input_files, mask_img, output_dir, labels=None,
                    regressor_files=None, regressor_names=None,
                    as_voxels=False, discard_scans=None,
                    n_jobs=1, **masker_kwargs):
    """Extract timeseries data from input files using an roi file to demark
    the region(s) of interest(s). This is the main function of this module.

    Parameters
    ----------
    input_files : list of niimg-like
        List of input NIfTI functional images
    mask_img : str
        Image that contains region mask(s). Can either be a single binary mask
        for a single region, or a numerically labeled atlas file. 0 must
        indicate background (non-region voxels).
    output_dir : str
        Save directory.
    labels : str or list of str
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
    discard_scans : int, optional
        The number of scans to discard at the start of each functional image,
        prior to any sort of extraction and post-processing. This is prevents
        unstabilized signals at the start from being included in signal
        standardization, etc.
    n_jobs : int, optional
        Number of processes to use for extraction if parallelization is
        dersired. Default is 1 (no parallelization)
    **masker_kwargs
        Keyword arguments for `nilearn.input_data` Masker objects.
    """
    mask_img = load_img(mask_img)
    masker = _set_masker(mask_img, as_voxels, **masker_kwargs)
    print(masker)

    # set as list of NoneType if no regressor files; makes it easy for
    # iterations
    if regressor_files is None:
        regressor_files = [regressor_files] * len(input_files)

    # no parallelization
    if n_jobs == 1:
        for i, img in enumerate(input_files):
            _mask_and_save(masker, img, output_dir, regressor_files[i],
                           regressor_names, as_voxels, labels, discard_scans)
    else:
        # repeat parameters are held constant for all parallelized iterations
        args = zip(
            repeat(masker),
            input_files, # iterate over
            repeat(output_dir),
            regressor_files, # iterate over, paired with input_files
            repeat(regressor_names),
            repeat(as_voxels),
            repeat(labels),
            repeat(discard_scans)
        )
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.starmap(_mask_and_save, args)

