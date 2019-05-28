"""Functions for command line interface
"""
import os
import argparse
import json
import glob
from natsort import natsorted

from niimasker.niimasker import extract_data

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, metavar='output_dir',
                        help='The path to the output directory')
    parser.add_argument('config', type=str.lower, metavar='config',
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


def _read_config(fname):
    """Read params and sort file inputs from configuration file"""
    with open(fname, 'rb') as f:
        conf = json.load(f)

    conf['input_files'] = _check_glob(conf['input_files'])
    if conf['regressor_files'] is not None:
        conf['regressor_files'] = _check_glob(conf['regressor_files'])
    return conf


def main():
    """Primary entrypoint in program"""
    opts = _cli_parser()
    params = _read_config(opts.config)

    params['output_dir'] = opts.output_dir

    # # append input directory
    # params['input_files'] = [os.path.join(opts.input_dir, i) for i in
    #                          params['input_files']]
    # params

    # if params['regressor_files'] is not None:
    #     params['regressor_files'] = [os.path.join(opts.input_dir, i) for i in
    #                                  params['regressor_files']]

    extract_data(**params)


if __name__ == '__main__':
    raise RuntimeError("`rextract/cli.py` should not be run directly. Please"
                       "`pip install` rextract and use the `rextract` command.")