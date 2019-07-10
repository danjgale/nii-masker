
import os
import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.datasets import fetch_atlas_aal
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker

from niimasker import niimasker


## HELPERS

def _get_atlas():
    """Fetch small atlas for testing purposes"""
    # save to path in test directory to only download atlas once
    test_path = os.path.dirname(__file__)
    data_dir = os.path.join(test_path, 'data-dir')
    os.makedirs(data_dir, exist_ok=True)
    return fetch_atlas_aal(data_dir=data_dir)


@pytest.fixture
def atlas_data():
    """Create a 4D image that repeats the atlas over 10 volumes.

    Every voxel/ROI is the same value across time. Therefore we can test if
    proper values are extracted (and if they are in the proper order).
    """
    atlas = _get_atlas()
    img = nib.load(atlas['maps'])
    return nib.concat_images([img] * 50)


@pytest.fixture
def regressors(tmpdir):
    """Mock regressors for confound regression testing"""
    reg_names = ['csf', 'wm', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y',
                 'rot_z']
    np.random.seed(42)
    data = np.random.rand(50, len(reg_names))
    return pd.DataFrame(data, columns=reg_names)


@pytest.fixture
def post_processed_data(atlas_data, regressors):
    """A post-processed version of the atlas_data, which is generated directly
    by nilearn rather than niimasker. The results from niimasker should
    directly match what is produced by nilearn.
    """
    labels_img = _get_atlas()['maps']
    masker = NiftiLabelsMasker(labels_img, standardize=True, smoothing_fwhm=5,
                               detrend=True, low_pass=.1, high_pass=.01, t_r=2)

    confounds = regressors.values
    return masker.fit_transform(atlas_data, confounds=confounds)


## TESTS


# def test_discard_initial_scans(atlas_data, regressors):
#     """Check if the correct number of scans are discarded at the start of the
#     image
#     """
#     # function works on the underlying numpy array not the dataframe
#     regressors = regressors.values

#     n_scans = 3
#     img, regs = niimasker._discard_initial_scans(atlas_data, n_scans, regressors)
#     assert img.get_data().shape[3] == atlas_data.get_data().shape[3] - n_scans
#     assert regs.shape[0] == regressors.shape[0] - n_scans


def test_set_masker(atlas_data):
    """Ensure that correct masker class is returned by _set_masker"""

    atlas = _get_atlas()

    # check multi ROI atlas
    atlas_img = nib.load(atlas['maps'])
    masker = niimasker._set_masker(atlas_img)
    assert isinstance(masker, NiftiLabelsMasker)

    # check single ROI atlas; create binary mask from atlas first
    bin_img = nib.Nifti1Image(np.where(atlas_img.get_data() == 2001, 1., 0),
                              atlas_img.affine)
    masker = niimasker._set_masker(bin_img)
    assert isinstance(masker, NiftiMasker)


def test_mask(atlas_data, regressors, post_processed_data):
    """Test basic (completely raw) and post-processed masking. Ensure results
    match equivalent versions created directly by nilearn"""

    atlas = _get_atlas()
    atlas_img = nib.load(atlas['maps'])

    # test basic mask with no post-processing
    test_masker = niimasker._set_masker(atlas_img)
    result = niimasker._mask(test_masker, atlas_data)

    expected_masker = NiftiLabelsMasker(atlas_img)
    expected = expected_masker.fit_transform(atlas_data)

    assert np.array_equal(result, expected)

    # test mask with all post-processing options
    test_masker = niimasker._set_masker(atlas_img, standardize=True,
                                        smoothing_fwhm=5, detrend=True,
                                        low_pass=.1, high_pass=.01, t_r=2)
    result = niimasker._mask(test_masker, atlas_data, confounds=regressors.values)

    assert np.array_equal(result, post_processed_data)











