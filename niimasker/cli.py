"""Functions for command line interface
"""
import sys
import os
import argparse
import json
import glob
import shutil
import pandas as pd
from natsort import natsorted
# import for version reporting
from platform import python_version
import nilearn
import nibabel
import scipy
import sklearn
import numpy
import natsort
import pkg_resources  # for niimasker itself

from niimasker.niimasker import make_timeseries
from niimasker.atlases import get_labelled_atlas

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, metavar='output_dir',
                        help='The path to the output directory')
    parser.add_argument('-i', '--input_files', nargs='+', type=str,
                        metavar='input_files',
                        help='One or more input NIfTI images. Can also be a '
                             'single string with a wildcard (*) to specify all '
                             'files matching the file pattern. If so, these '
                             'files are naturally sorted by file name prior to '
                             'extraction.')
    parser.add_argument('-r', '--roi_file', type=str, metavar='roi_file', 
                        help='Parameter that defines the region(s) of interest. '
                             'This can be 1) a file path to NIfTI image that is '
                             'an atlas of multiple regions or a binary mask of '
                             'one region, 2) a nilearn query string formatted as '
                             '`nilearn:<atlas-name>:<atlas-parameters> (see '
                             'online documentation), or 3) a file path to a '
                             '.tsv file that has x, y, z columns that contain '
                             'roi_file coordinates in MNI space. Refer to online '
                             'documentation for more on how these options map '
                             'onto the underlying nilearn masker classes.')
    parser.add_argument('-m', '--mask_img', type=str, metavar='mask_img',
                        help='File path of a NIfTI mask image a to be used when '
                             '`roi_file` is a) an multi-region atlas or a b) list '
                             'of coordinates. This will restrict extraction to '
                             'only voxels within the mask. If `roi_file` is a '
                             'single-region binary mask, this will be ignored.')
    parser.add_argument('--labels', nargs='+', type=str, metavar='labels',
                        help='Labels corresponding to the mask numbers in '
                             '`mask`. Can either be a list of strings, or a '
                             '.tsv file that contains a `Labels` column. Labels '
                             'must be sorted in ascending order to correctly '
                             'correspond to the atlas indices. The number of '
                             'labels provided must match the number of non-zero '
                             'indices in `mask`. If none are provided, numeric '
                             'indices are used (default)')
    parser.add_argument('--regressor_files', nargs='+', type=str,
                        metavar='regressor_files',
                        help='One or more tabular files with regressors in each '
                             'column. The number of files match the number of '
                             'input NIfTI files provided and must be in the '
                             'same order. The number of rows in each file must '
                             'match the number of timepoints in their '
                             'respective input NIfTI files. Can also be a '
                             'single string with a wildcard (*) to specify all '
                             'files matching the file pattern. If so, these '
                             'files are naturally sorted by file name prior to '
                             'extraction. Double check to make sure these are '
                             'correctly aligned with the input NIfTI files.')
    parser.add_argument('--regressor_names', nargs='+', type=str,
                        metavar='regressor_names',
                        help='The regressor names to use for confound '
                             'regression. Applies to all regressor files and '
                             'the names must correspond to headers in each '
                             'file. If no regressor names are provided, but '
                             'files are, all regressors in regressor files '
                             'are used.')
    parser.add_argument('--denoising_strategy', nargs='+',type=str,
                        metavar='denoising_strategy',
                        help='The denoising strategy to use for confound '
                             'regression. Applies to all regressor files. '
                             'The denoising strategy must be either one predefined by '
                             'load_confounds or a list compatible with load_confounds flexible '
                             'denoising strategy options. See the documentation '
                             ' https://github.com/SIMEXP/load_confounds. If no denoising strategy is provided, '
                             'but files are, all regressors in regressor files '
                             'are used.')
    parser.add_argument('--as_voxels', default=False,
                        action='store_true',
                        help='Whether to extract out the timeseries of each '
                             'voxel instead of the mean timeseries. This is '
                             'only available for single ROI binary masks. '
                             'Default False.')
    parser.add_argument('--radius', type=float, metavar='radius', 
                        help='Set the radius of the spheres (in mm) centered on '
                             'the coordinates provided in `roi_file`. Only applicable '
                             'when a coordinate .tsv file is passed to `roi_file`; '
                             'otherwise, this will be ignored. If not set, '
                             'the nilearn default of extracting from a single '
                             'voxel (the coordinates) will be used.')
    parser.add_argument('--allow_overlap', action='store_true', default=False,
                        help='Permit overlapping spheres when coordinates are '
                             'provided to `roi_file` and sphere-radius is not None.')                               
    parser.add_argument('--standardize',
                        action='store_true', default=False,
                        help='Whether to standardize (z-score) each timeseries. '
                             'Default False')
    parser.add_argument('--t_r', type=int, metavar='t_r',
                        help='The TR of the input NIfTI files, specified in '
                             'seconds. Must be included if temporal filtering '
                             'or realignment derivatives are specified.')
    parser.add_argument('--high_pass', type=float, metavar='high_pass',
                        help='High pass filter cut off in Hertz. If it is not '
                             'specified, no filtering is done. (default)')
    parser.add_argument('--low_pass', type=float, metavar='low_pass',
                        help='Low pass filter cut off in Hertz.')
    parser.add_argument('--detrend', action='store_true',
                        default=False,
                        help='Whether to temporally detrend the data.')
    parser.add_argument('--smoothing_fwhm', type=float, metavar='smoothing_fwhm',
                        help='Smoothing kernel FWHM (in mm) if spatial smoothing '
                             'is desired.')
    parser.add_argument('--discard_scans', type=int, metavar='discard_scans',
                        help='Discard the first N scans of each functional '
                             'NIfTI image.')
    parser.add_argument('--n_jobs', type=int, metavar='n_jobs', default=1,
                        help='The number of CPUs to use if parallelization is '
                             'desired. Default is 1 (serial processing).')
    parser.add_argument('-c', '--config', type=str.lower, metavar='config',
                        help='Configuration .json file as an alternative to '
                             'command-line arguments. See online documentation '
                             'for what keys to include.')
    return parser.parse_args()


def _check_glob(x):
    if isinstance(x, str):
        return natsorted(glob.glob(x))
    elif isinstance(x, list):
        return x
    else:
        raise ValueError('Input data files (NIfTIs and confounds) must be a'
                         'string or list of string')

def _empty_to_None(x):
    """Replace an empty list from params with None"""
    if isinstance(x, list):
        if not x:
            x = None
    return x

def _check_params(params):
    """Ensure that required fields are included and correctly formatted"""

    # convert empty list parameters to None 
    params['labels'] = _empty_to_None(params['labels'])
    params['denoising_strategy'] = _empty_to_None(params['denoising_strategy'])
    params['regressor_names'] = _empty_to_None(params['regressor_names'])
    params['regressor_files'] = _empty_to_None(params['regressor_files'])

    if params['input_files'] is None:
        raise ValueError('Missing input files. Check files')
    else:
        params['input_files'] = _check_glob(params['input_files'])
        # glob returned nothing
        if not params['input_files']:
            raise ValueError('Missing input files. Check files')

    if not params['roi_file']:
        raise ValueError('Missing roi_file input.')

    if params['regressor_files'] is not None:
        params['regressor_files'] = _check_glob(params['regressor_files'])

    if isinstance(params['labels'], str):
        if params['labels'].endswith('.tsv'):
            df = pd.read_table(params['labels'])
            params['labels'] = df['Label'].tolist()
        else:
            raise ValueError('Labels must be a filename or a list of strings.')

    if params['roi_file'].startswith('nilearn:'):
        cache = os.path.join(params['output_dir'], 'niimasker_data')
        os.makedirs(cache, exist_ok=True)
        atlas, labels = get_labelled_atlas(params['roi_file'], data_dir=cache,
                                           return_labels=True)
        params['roi_file'] = atlas
        params['labels'] = labels

    return params


def _merge_params(cli, config):
    """Merge CLI params with configuration file params. Note that the
    configuration params will overwrite the CLI params."""

    # update CLI params with configuration; overwrites
    params = dict(list(cli.items()) + list(config.items()))
    return params

def main():
    """Primary entrypoint in program"""
    params = vars(_cli_parser())

    # read config file if available
    if params['config'] is not None:
        with open(params['config'], 'rb') as f:
            conf_params = json.load(f)
        params = _merge_params(params, conf_params)

    params.pop('config')

    # finalize parameters
    os.makedirs(params['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(params['output_dir'], 'niimasker_data'),
                exist_ok=True)
    params = _check_params(params)

    # add in meta data
    versions = {
        'python': python_version(),
        'niimasker': pkg_resources.require("niimasker")[0].version,
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
        'nilearn': nilearn.__version__,
        'nibabel': nibabel.__version__,
        'natsort': natsort.__version__
        #load_confounds version?
    }

    # export command-line call and parameters to a file
    param_info = {'command': " ".join(sys.argv), 'parameters': params,
                  'meta_data': versions}

    metadata_path = os.path.join(params['output_dir'], 'niimasker_data')
    param_file = os.path.join(metadata_path, 'parameters.json')

    with open(param_file, 'w') as fp:
        json.dump(param_info, fp, indent=2)

    shutil.copy2(params['roi_file'], metadata_path)

    # run extraction
    make_timeseries(**params)


if __name__ == '__main__':
    raise RuntimeError("`niimasker/cli.py` should not be run directly. Please "
                       "`pip install` rextract and use the `rextract` command.")
