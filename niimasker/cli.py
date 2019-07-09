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
from niimasker.plots import make_figures

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
    parser.add_argument('-m', '--mask_img', type=str, metavar='mask_img',
                        help='File path of the atlas/ROI mask. Can either be a '
                             'single ROI mask that is binary, or an atlas with '
                             'numeric labels. Must be a sinlge NIfTI file in '
                             'the same space as the input images.')
    parser.add_argument('--labels', nargs='+', type=str, metavar='labels',
                        help='Labels corresponding to the mask numbers in '
                             '`mask`. They must be sorted in ascending order '
                             'to correctly correspond to the atlas indices. The '
                             'number of labels provided must match the number '
                             'of non-zero indices in `mask`. If none are '
                             'provided, numeric indices are used')
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
    parser.add_argument('--realign_derivs',
                        default=False, action='store_true',
                        help='Whether to include temporal derivatives of '
                             'realignment regressors. --t_r must be specified.')
    parser.add_argument('--as_voxels', default=False,
                        action='store_true',
                        help='Whether to extract out the timeseries of each '
                             'voxel instead of the mean timeseries. This is '
                             'only available for single ROI binary masks. '
                             'Default False.')
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
                             'specified, no filtering is done.')
    parser.add_argument('--low_pass', type=float, metavar='low_pass',
                        help='Low pass filter cut off in Hertz. If it is not '
                             'specified, no filtering is done.')
    parser.add_argument('--detrend', action='store_true',
                        default=False,
                        help='Whether to detrend the data. Default False')
    parser.add_argument('--smoothing_fwhm', type=float, metavar='smoothing_fwhm',
                        help='Smoothing kernel FWHM (in mm) if spatial smoothing '
                             'is desired. If not specified, no smoothing is '
                             'performed.')
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


def _check_params(params):
    """Ensure that required fields are included and correctly formatted"""

    if params['input_files'] is None:
        raise ValueError('Missing input files. Check files')
    else:
        params['input_files'] = _check_glob(params['input_files'])

    if not params['input_files']:
        raise ValueError('Missing input files. Check files')

    if not params['mask_img']:
        raise ValueError('Missing mask file.')

    if params['regressor_files'] is not None:
        params['regressor_files'] = _check_glob(params['regressor_files'])

    if isinstance(params['labels'], str):
        if params['labels'].endswith('.csv'):
            df = pd.read_csv(params['labels'])
            params['labels'] = df['Label'].tolist()

    if params['mask_img'].startswith('nilearn:'):
        cache = os.path.join(params['output_dir'], 'niimasker_data')
        os.makedirs(cache, exist_ok=True)
        atlas, labels = get_labelled_atlas(params['mask_img'], data_dir=cache,
                                           return_labels=True)
        params['mask_img'] = atlas
        params['labels'] = labels

    return params


def _merge_params(cli, config):
    """Merge CLI params with configuration file params. Note that the
    configuration params will overwrite the CLI params."""

    # update CLI params with configuration; overwrites
    params = dict(list(cli.items()) + list(config.items()))

    params.pop('config')
    return params


def main():
    """Primary entrypoint in program"""
    params = vars(_cli_parser())

    # read config file if available
    if params['config'] is not None:
        with open(params['config'], 'rb') as f:
            conf_params = json.load(f)
        params = _merge_params(params, conf_params)
    else:
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
    }

    # export command-line call and parameters to a file
    param_info = {'command': " ".join(sys.argv), 'parameters': params,
                  'meta_data': versions}

    metadata_path = os.path.join(params['output_dir'], 'niimasker_data')
    param_file = os.path.join(metadata_path, 'parameters.json')
    print(param_info)
    with open(param_file, 'w') as fp:
        json.dump(param_info, fp, indent=2)

    shutil.copy2(params['mask_img'], metadata_path)

    # run extraction
    make_timeseries(**params)

    print('  Making reports...')
    # generate figures and report
    if params['as_voxels']:
        make_figures(params, as_carpet=True, connectivity_metrics=False)
    else:
        make_figures(params)


if __name__ == '__main__':
    raise RuntimeError("`niimasker/cli.py` should not be run directly. Please "
                       "`pip install` rextract and use the `rextract` command.")
