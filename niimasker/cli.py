"""Functions for command line interface
"""

import argparse
import json
import glob
import pandas as pd
from natsort import natsorted

from niimasker.niimasker import extract_data

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, metavar='output_dir',
                        help='The path to the output directory')
    parser.add_argument('-i', '--input_files', nargs='+', type=str,
                        metavar='input_files',
                        help='The path to the output directory')
    parser.add_argument('-m', '--mask', type=str, metavar='mask',
                        help='The path to the output directory')
    parser.add_argument('--labels', nargs='+', type=str, metavar='labels',
                        help='The path to the output directory')
    parser.add_argument('--regressor_files', nargs='+', type=str,
                        metavar='regressor_files',
                        help='The path to the output directory')
    parser.add_argument('--regressor_names', nargs='+', type=str,
                        metavar='regressor_names',
                        help='The path to the output directory')
    parser.add_argument('--as_voxels', type=bool, metavar='as_voxels',
                        default=False, help='The path to the output directory')
    parser.add_argument('--standardize', type=bool, metavar='standardize',
                        default=False, help='The path to the output directory')
    parser.add_argument('--t_r', type=int, metavar='t_r',
                        help='The path to the output directory')
    parser.add_argument('--high_pass', type=float, metavar='high_pass',
                        help='The path to the output directory')
    parser.add_argument('--low_pass', type=float, metavar='low_pass',
                        help='The path to the output directory')
    parser.add_argument('--detrend', type=bool, metavar='detrend',
                        default=False, help='The path to the output directory')
    parser.add_argument('--smoothing_fwhm', type=float, metavar='smoothing_fwhm',
                        help='The path to the output directory')
    parser.add_argument('--discard_scans', type=int, metavar='discard_scans',
                        help='The path to the output directory')
    parser.add_argument('-c', '--config', type=str.lower, metavar='config',
                        help='Configuration .json file for ROI extraction. See'
                                'documentation for what keys to include.')
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

    if 'input_files' not in params:
        raise ValueError('Missing input file(s) (flag -i)')
    else:
        params['input_files'] = _check_glob(params['input_files'])

    if 'mask' not in params:
        raise ValueError('Missing mask file (flag -m)')

    if params['regressor_files'] is not None:
        params['regressor_files'] = _check_glob(params['regressor_files'])

    if isinstance(params['labels'], str) & params['labels'].endswith('.csv'):
        df = pd.read_csv(params['labels'])
        params['labels'] = df['Label'].tolist()

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

    if 'config' in params:
        with open(params['config'], 'rb') as f:
            conf_params = json.load(f)
        params = _merge_params(params, conf_params)

    params = _check_params(params)
    # display
    print('INPUT PARAMETERS:')
    for k, v in params.items():
        print('  {}: {}'.format(k, v))

    print('RUNNING:')
    extract_data(**params)


if __name__ == '__main__':
    raise RuntimeError("`niimasker/cli.py` should not be run directly. Please"
                       "`pip install` rextract and use the `rextract` command.")