# -*- coding: utf-8 -*-

"""SF DECONVOLVE ARGUMENTS

This module sets the arguments for sf_deconvolve.py.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 2.4

:Date: 23/10/2017

"""

from __future__ import absolute_import
import argparse as ap
from argparse import ArgumentDefaultsHelpFormatter as formatter


class ArgParser(ap.ArgumentParser):

    """Argument Parser

    This class defines a custom argument parser to override the
    default convert_arg_line_to_args method from argparse.

    """

    def __init__(self, *args, **kwargs):

        super(ArgParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        """Convert argument line to arguments

        This method overrides the default method of argparse. It skips blank
        and comment lines, and allows .ini style formatting.

        Parameters
        ----------
        line : str
            Input argument string

        Yields
        ------
        str
            Argument strings

        """

        line = line.split()
        if line and line[0][0] not in ('#', ';'):
            if line[0][0] != '-':
                line[0] = '--' + line[0]
            if len(line) > 1 and '=' in line[0]:
                line = line[0].split('=') + line[1:]
            for arg in line:
                yield arg

def get_opts(args=None):
    """
    Get script options for PSF Deconvolution.

    This method sets up the argument parser for the PSF Deconvolution script,
    handling both required and optional arguments. It includes options for 
    input file names, dictionary size, image size, iterations, and various 
    parameters relevant to the deconvolution process.

    Parameters
    ----------
    args : list, optional
        A list of arguments to parse (default is None, which uses sys.argv).

    Returns
    -------
    argparse.Namespace
        An object containing the parsed arguments.
    """

    # Set up argument parser
    parser = ArgParser(
        add_help=False,
        usage='%(prog)s [options]',
        description='PSF Deconvolution Script',
        formatter_class=formatter,
        fromfile_prefix_chars='@'
    )

    # Required and Optional Argument Groups
    required = parser.add_argument_group('Required Arguments')
    optional = parser.add_argument_group('Optional Arguments')

    # Required Arguments with Default Values
    required.add_argument(
        '-ih', '--inputhigh',
        default='datasamples_new/input_hrd.csv',
        help='Input data file name (high resolution). Default: "datasamples_new/input_hrd.csv"'
    )

    required.add_argument(
        '-il', '--inputlow',
        default='datasamples_new/input_lrd.csv',
        help='Input data file name (low resolution). Default: "datasamples_new/input_lrd.csv"'
    )

    required.add_argument(
        '-d', '--dictsize', type=int, default=100,
        help='Dictionary size. Default: 100'
    )

    required.add_argument(
        '-img', '--imageN', type=int, default=2500,
        help='Size of input image. Default: 2500'
    )

    # Optional Arguments
    optional.add_argument(
        '-h', '--help', action='help',
        help='Show this help message and exit.'
    )

    optional.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress verbose output.'
    )

    optional.add_argument(
        '-n', '--n_iter', type=int, default=150,
        help='Number of iterations. Default: 150'
    )

    # More Optional Arguments
    optional.add_argument('--window', type=int, default=10, help='Window size to measure error. Default: 10')
    optional.add_argument('--bands_h', type=int, default=200, help='Number of bands in high resolution. Default: 200')
    optional.add_argument('--bands_l', type=int, default=100, help='Number of bands in low resolution. Default: 100')
    optional.add_argument('--c1', type=float, default=0.4, help='Parameter c1. Default: 0.4')
    optional.add_argument('--c2', type=float, default=0.4, help='Parameter c2. Default: 0.4')
    optional.add_argument('--c3', type=float, default=0.8, help='Parameter c3. Default: 0.8')
    optional.add_argument('--maxbeta', type=float, default=1e+6, help='Maximum beta value. Default: 1e+6')
    optional.add_argument('--delta', type=float, default=1e-4, help='Delta value. Default: 1e-4')
    optional.add_argument('--beta', type=float, default=0.01, help='Beta value. Default: 0.01')
    optional.add_argument('--lamda', type=float, default=0.1, help='Lambda value. Default: 0.1')

    # Return the parsed arguments
    return parser.parse_args(args)