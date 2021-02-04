"""Core module that contains all functions related to extracting out time
series data.
"""

import os
import warnings
import multiprocessing
import load_confounds
import numpy as np
import pandas as pd
import nibabel as nib
from itertools import repeat
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.input_data import (NiftiMasker, NiftiLabelsMasker,NiftiSpheresMasker)
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from niimasker.report import generate_report


class FunctionalImage(object):
    def __init__(self, fname):

        self.fname = fname
        img = nib.load(self.fname)
        self.img = img

        self.regressors = None
        self.regressor_file = None
                    
    def set_regressors(self, regressor_fname, denoiser = None):
        """Use denoiser to set appropriate regressors."""
        self.regressor_file = regressor_fname
        
        #No strategy passed
        if denoiser is None:
            if regressor_fname is None:
                pass #do nothing
            else: #If there is a file nothing specified, use whole file
                self.regressors = pd.read_csv(self.regressor_file, sep=r'\t',engine='python').values
        elif isinstance(denoiser,list): #denoiser is list of str (regressor_names)
            self.regressors = pd.read_csv(self.regressor_file, sep=r'\t',engine='python')[denoiser].values
        else: #denoiser is a load_confounds parser
            self.regressors = denoiser.load(self.regressor_file)
            
                    
    def discard_scans(self, n_scans):
        # crop scans from image
        arr = self.img.get_data()
        arr = arr[:, :, :, n_scans:]
        self.img = nib.Nifti1Image(arr, self.img.affine)

        if self.regressors is not None:
            # crop from regressors
            self.regressors = self.regressors[n_scans:, :]


    def extract(self, masker, as_voxels=False, labels=None):
        print('  Extracting from {}'.format(os.path.basename(self.fname)))
        
        timeseries = masker.fit_transform(self.img,confounds=self.regressors)

        # determine column names for timeseries
        if isinstance(masker, NiftiMasker):
            labels = ['voxel {}'.format(int(i))
                      for i in np.arange(timeseries.shape[1])]
            self.roi_img = masker.mask_img_
            self.masker_type = 'NiftiMasker'
            
        elif isinstance(masker, NiftiLabelsMasker):
            if labels is None:
                labels = ['roi {}'.format(int(i)) for i in masker.labels_]
            self.roi_img = masker.labels_img
            self.masker_type = 'NiftiLabelsMasker'

        elif isinstance(masker, NiftiSpheresMasker):
            if labels is None:
                labels = ['roi {}'.format(int(i)) for i in range(len(masker.seeds))]
            self.roi_img = masker.spheres_img
            self.masker_type = 'NiftiSpheresMasker'

        self.masker = masker
        self.data = pd.DataFrame(timeseries, columns=[str(i) for i in labels])


## MASKING FUNCTIONS
def _get_spheres_from_masker(masker, img):
    """Re-extract spheres from coordinates to make niimg. 
    Note that this will take a while, as it uses the exact same function that
    nilearn calls to extract data for NiftiSpheresMasker
    """

    ref_img = nib.load(img) 
    ref_img = nib.Nifti1Image(ref_img.get_fdata()[:, :, :, [0]], ref_img.affine)

    X, A = _apply_mask_and_get_affinity(masker.seeds, ref_img, masker.radius, 
                                        masker.allow_overlap)
    # label sphere masks
    spheres = A.toarray()
    spheres *= np.arange(1, len(masker.seeds) + 1)[:, np.newaxis]

    # combine masks, taking the maximum if overlap occurs
    arr = np.zeros(spheres.shape[1])
    for i in np.arange(spheres.shape[0]):
        arr = np.maximum(arr, spheres[i, :])
    arr = arr.reshape(ref_img.shape[:-1])
    spheres_img = nib.Nifti1Image(arr, ref_img.affine)
    
    if masker.mask_img is not None:
        mask_img_ = resample_to_img(masker.mask_img, spheres_img)
        spheres_img = math_img('img1 * img2', img1=spheres_img, 
                               img2=mask_img_)

    return spheres_img


def _read_coords(roi_file):
    """Parse and validate coordinates from file"""

    if not roi_file.endswith('.tsv'):
        raise ValueError('Coordinate file must be a tab-separated .tsv file')

    coords = pd.read_table(roi_file)
    
    # validate columns
    columns = [x for x in coords.columns if x in ['x', 'y', 'z']]
    if (len(columns) != 3) or (len(np.unique(columns)) != 3):
        raise ValueError('Provided coordinates do not have 3 columns with '
                         'names `x`, `y`, and `z`')

    # convert to list of lists for nilearn input
    return coords.values.tolist()


def _set_masker(roi_file, as_voxels=False, **kwargs):
    """Check and see if multiple ROIs exist in atlas file"""

    if isinstance(roi_file, str) and roi_file.endswith('.tsv'):
        roi = _read_coords(roi_file)
        n_rois = len(roi)
        is_coords = True
        print('  {} region(s) detected from coordinates'.format(n_rois))
    else:
        roi = load_img(roi_file)
        n_rois = len(np.unique(roi.get_data())) - 1

        is_coords = False
        print('  {} region(s) detected from {}'.format(n_rois,
                                                       roi.get_filename()))
    
    if is_coords:
        if kwargs.get('radius') is None:
            warnings.warn('No radius specified for coordinates; setting '
                            'to nilearn.input_data.NiftiSphereMasker default '
                            'of extracting from a single voxel')
        masker = NiftiSpheresMasker(roi, **kwargs)
    else:

        if 'radius' in kwargs:
            kwargs.pop('radius')
        
        if 'allow_overlap' in kwargs:
            kwargs.pop('allow_overlap')
        
        if n_rois > 1:
            masker = NiftiLabelsMasker(roi, **kwargs)
        elif n_rois == 1:
            # single binary ROI mask 
            if as_voxels:
                if 'mask_img' in kwargs:
                    kwargs.pop('mask_img')
                masker = NiftiMasker(roi, **kwargs)
            else:
                # more computationally efficient if only wanting the mean of ROI
                masker = NiftiLabelsMasker(roi, **kwargs)
        else:
            raise ValueError('No ROI detected; check ROI file')
    
    return masker
                
def _make_denoiser(denoising_strategy):
    """Parses denoising_strategy and creates load_confounds object."""
    #predefined strategy
    if (len(denoising_strategy) == 1) and (denoising_strategy[0] in ['Params2','Params6','Params9','Params24','Params36','AnatCompCor','TempCompCor']):
        denoiser = eval('load_confounds.{}()'.format(denoising_strategy[0]))
    #flexible strategy
    elif set(denoising_strategy) <= set(['motion','high_pass','wm_csf', 'compcor', 'global']):
        denoiser = load_confounds.Confounds(strategy=denoising_strategy)
    else:
        raise ValueError('Provided denoising strategy is not recognized.')
    return denoiser

def _set_denoiser(denoising_strategy=None,regressor_names=None,regressor_files=None):
    """Handles denoising arguments, returns either None, load_confounds object, or regressor_names."""
    #if there are files provided
    if regressor_files is not None:
        if regressor_names is not None:
            if denoising_strategy is not None:
                warnings.warn('Both regressor_names and denoising_strategy were specified, only regressor_names will be used.')
            denoiser = regressor_names #list of str
            
        elif denoising_strategy is not None:
            denoiser = _make_denoiser(denoising_strategy) #load_confounds object
        else:
            warnings.warn('No strategy specified, full regressor_files will be used for denoising.')
            denoiser = None
    #If there are no files
    else:
        if (regressor_names is not None) or (denoising_strategy is not None):
            warnings.warn('Either regressor_names or denoising_strategy were specified without regressor_files. '
                          'Continuing without denoising.')
        denoiser = None
        #should set regressor_names and denoising_strategy to None here?
    return denoiser


def _mask_and_save(masker, denoiser, img_name, output_dir, regressor_file=None,
                  as_voxels=False,labels=None, discard_scans=None):
    """Runs the full masking process and saves output for a single image;
    the main function used by `make_timeseries`
    """
    img = FunctionalImage(img_name)
    
    img.set_regressors(regressor_file, denoiser)
        
    if discard_scans is not None:
        if discard_scans > 0:
            img.discard_scans(discard_scans)

    img.extract(masker, as_voxels=as_voxels, labels=labels)

    # export data and report
    out_fname = os.path.basename(img.fname).split('.')[0] + '_timeseries.tsv'
    img.data.to_csv(os.path.join(output_dir, out_fname), sep='\t', index=False,
                    float_format='%.8f')
    generate_report(img, output_dir)


def make_timeseries(input_files, roi_file, output_dir, labels=None,
                    regressor_files=None, denoising_strategy=None, regressor_names=None,
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
    denoising_strategy: str or list of str, optional
        Specifies which load_confounds parser to use.
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
    masker = _set_masker(roi_file, as_voxels, **masker_kwargs)
    denoiser = _set_denoiser(denoising_strategy=denoising_strategy,regressor_names=regressor_names,regressor_files=regressor_files)

    # create and save spheres image if coordinates are provided
    if isinstance(masker, NiftiSpheresMasker):
        masker.spheres_img = _get_spheres_from_masker(masker, input_files[0])
        masker.spheres_img.to_filename(os.path.join(output_dir, 'niimasker_data','spheres_image.nii.gz'))

    # set as list of NoneType if no regressor files; makes it easy for
    # iterations
    if regressor_files is None:
        regressor_files = [regressor_files] * len(input_files)

    # no parallelization
    if n_jobs == 1:
        for i, img in enumerate(input_files):
            _mask_and_save(masker, denoiser, img, output_dir, regressor_files[i],
                            as_voxels, labels, discard_scans)
    else:
        # repeat parameters are held constant for all parallelized iterations
        args = zip(
            repeat(masker),
            input_files, # iterate over
            repeat(output_dir),
            regressor_files, # iterate over, paired with input_files
            repeat(denoising_strategy),
            repeat(as_voxels),
            repeat(labels),
            repeat(discard_scans)
        )
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.starmap(_mask_and_save, args)

