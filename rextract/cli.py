"""
Functions for command line interface to generate cluster / peak summary
"""
import argparse
import os.path as op

from rextract.rextract import extract_data

def _rextract_parser():
    """ Reads command line arguments and returns input specifications """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str.lower, metavar='config',
                        help='Configuration .json file for ROI extraction. See'
                             'documentation for what keys to include.')
    return parser.parse_args()


def _read_config(fname):
    """Read and interpret input from config .json file"""
    pass


def main():
    """Primary entrypoint in program"""
    opts = _rextract_parser()
    config = _read_config(config)
    extract_data()


if __name__ == '__main__':
    raise RuntimeError("`rextract/cli.py` should not be run directly. Please"
                       "`pip install` rextract and use the `rextract` command.")