# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""Parser for utils."""
import argparse


def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict
    Notes
    -----
    # Argument parser follow template provided by RalphyZ.
    # https://stackoverflow.com/a/43456577
    """
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('Required Argument:')

    required.add_argument('-indir', '--input-directory',
                          dest='root',
                          type=str,
                          help='Specify root directory of dataset',
                          default=None)
    required.add_argument('-sub', '--subject',
                          dest='sub',
                          type=str,
                          help='Specify alongside \"-heur\". Code of '
                               'subject to process.',
                          default=None)
    optional.add_argument('-ses', '--session',
                          dest='ses',
                          type=str,
                          help='Specify alongside \"-heur\". Code of '
                               'session to process.',
                          default=None)
    optional.add_argument('-type', '--file-type',
                          dest='type',
                          type=str,
                          help='Specify what file type you want to list\n'
                               'Defaults to .acq files',
                          default='.acq')
    optional.add_argument('-show', '--show-dict',
                          dest='show',
                          help='Specify if you want to print dictionary',
                          default=False)
    optional.add_argument('-save', '--save-dict',
                          dest='save',
                          help='Specify if you want to save the dictionary',
                          default=False)
    parser._action_groups.append(optional)

    return parser


def _get_parser2():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict
    Notes
    -----
    # Argument parser follow template provided by RalphyZ.
    # https://stackoverflow.com/a/43456577
    """
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('Required Argument:')

    required.add_argument('-indir', '--input-directory',
                          dest='root',
                          type=str,
                          help='Specify root directory of dataset',
                          default=None)
    required.add_argument('-sub', '--subject',
                          dest='sub',
                          type=str,
                          help='Specify alongside \"-heur\". Code of '
                               'subject to process.',
                          default=None)
    optional.add_argument('-ses', '--session',
                          dest='ses',
                          type=str,
                          help='Specify alongside \"-heur\". Code of '
                               'session to process.',
                          default=None)

    optional.add_argument('-show', '--show-dict',
                          dest='show',
                          help='Specify if you want to print dictionary',
                          default=False)
    optional.add_argument('-save', '--save-dict',
                          dest='save',
                          help='Specify if you want to save the dictionary',
                          default=False)
    parser._action_groups.append(optional)

    return parser


if __name__ == '__main__':
    raise RuntimeError('CLI.py should not be run directly')
