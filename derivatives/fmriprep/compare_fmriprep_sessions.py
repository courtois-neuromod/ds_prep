#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Compare two fmriPrep session and output a short report
"""


import argparse
import filecmp
import json
import logging
import nibabel as nib
from nilearn.input_data import MultiNiftiMasker
import numpy as np
from scipy.spatial.distance import dice
import os

import pandas as pd

templates = {
        'boldref': '_boldref.nii.gz',
        'confound': '_desc-confounds_regressors.tsv',
        'mask': '_desc-brain_mask.nii.gz',
        'preproc': '_desc-preproc_bold.nii.gz'
}


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_sess1',
                   help='Path to the first fMRIPrep session.')
    p.add_argument('in_sess2',
                   help='Path to the second fMRIPrep session.')
    p.add_argument('out_file',
                   help='output tsv file.')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    p.add_argument('-v', action='store_true', dest='verbose',
                   help='If set, produces verbose output.')
    return p


def assert_fMRIPrep_structure(parser, list_sess):
    """
    Assert if paths are real fMRIPrep sessions and return dircmp object

    Parameters
    -------
    parser : parser
    list_sess:  list
        List of two sessions to be compared

    Returns
    -------
    cmp : dircmp
        Comparaison between the two sessions
    """
    for curr_sess in list_sess:
        curr_folders = os.listdir(curr_sess)
        if not 'func' in curr_folders:
            parser.error('{} does not seem to be a fMRIPrep '
                         'folder'.format(curr_sess))

    return filecmp.dircmp(list_sess[0], list_sess[1])


def filter_files(diff_files):
    """
    Assert if paths are real fMRIPrep sessions and return dircmp object

    Parameters
    -------
    parser : parser
    list_sess:  list
        List of two sessions to be compared

    Returns
    -------
    cmp : dircmp
        Comparaison between the two sessions
    """
    res = {
        'boldref': [],
        'confound': [],
        'mask': [],
        'preproc': []
    }
    for nKey in templates.keys():
        res[nKey] = [i for i in diff_files if templates[nKey] in i]

    return res


def compare_files(curr_type, in_files, sessions):
    """
    Assert if paths are real fMRIPrep sessions and return dircmp object

    Parameters
    -------
    parser : parser
    list_sess:  list
        List of two sessions to be compared

    Returns
    -------
    cmp : dircmp
        Comparaison between the two sessions
    """
    curr_json = []
    if curr_type == 'confound':
        for curr_file in in_files:
            tmp_json = compare_confound(curr_file, curr_type, sessions)
            curr_json = curr_json + tmp_json
    else:
        for curr_file in in_files:
            tmp_json = compare_nifti(curr_file, curr_type, sessions)
            curr_json = curr_json + tmp_json

    return curr_json


def compare_nifti(curr_file, type_file, sessions):
    """
    Assert if paths are real fMRIPrep sessions and return dircmp object

    Parameters
    -------

    Returns
    -------
    o_dict: dict

    """

    in_path_1 = os.path.join(sessions[0], 'func', curr_file)
    in_path_2 = os.path.join(sessions[1], 'func', curr_file)

    in_img_1 = nib.load(in_path_1)
    in_img_2 = nib.load(in_path_2)
    if type_file != 'mask':
        in_data_1 = in_img_1.get_fdata(dtype=np.float32)
        in_data_2 = in_img_2.get_fdata(dtype=np.float32)

        masker = MultiNiftiMasker()
        time_series = masker.fit_transform([in_img_1, in_img_2])

        correlations = []
        for nTime in range(time_series[0].shape[0]):
            correlations.append(np.corrcoef(time_series[0][nTime],
                                            time_series[1][nTime])[1, 0])

        diff = np.abs(time_series[0] - time_series[1])

        if len(correlations) == 1:
            corrValues = correlations
        else:
            corrValues = {'min': float(np.min(correlations)),
                          'max': float(np.max(correlations)),
                          'mean': float(np.mean(correlations)),
                          'med': float(np.median(correlations)),
                          'std': float(np.std(correlations))}

        diffValues = {'min': float(np.min(diff)),
                      'max': float(np.max(diff)),
                      'mean': float(np.mean(diff)),
                      'med': float(np.median(diff)),
                      'std': float(np.std(diff))}

        return [{'filename': curr_file,
                 'type': type_file,
                 'correlation': corrValues,
                 'abs_diff': diffValues}]

    else:
        in_data_1 = np.asanyarray(in_img_1.dataobj).astype(np.bool)
        in_data_2 = np.asanyarray(in_img_2.dataobj).astype(np.bool)

        diceValue = dice(in_data_1.reshape(-1), in_data_2.reshape(-1))
        return [{'filename': curr_file,
                 'type': type_file,
                 'dice': float(diceValue)}]


def compare_confound(curr_file, type_file, sessions):

    in_path_1 = os.path.join(sessions[0], 'func', curr_file)
    in_path_2 = os.path.join(sessions[1], 'func', curr_file)

    confounds_1 = pd.read_csv(in_path_1, delimiter="\t", encoding="utf-8")
    confounds_2 = pd.read_csv(in_path_2, delimiter="\t", encoding="utf-8")

    if np.all(len(confounds_1.columns) != len(confounds_2.columns)):
        set_1 = set(confounds_1.columns.values)
        set_2 = set(confounds_2.columns.values)
        diff_columns = list(set_1-set_2)
        common_columns = list(set_1.intersection(set_2))
    else:
        common_columns = list(confound_1.columns.values)
        diff_columns = []

    correlations = []
    for curr_column in common_columns:
        correlations.append(confounds_1[curr_column].corr(confounds_2[curr_column]))

    corrValues = {'min': {"value": float(correlations[np.argmin(correlations)]),
                          "column": common_columns[np.argmin(correlations)]},
                  'max': {"value": float(correlations[np.argmax(correlations)]),
                          "column": common_columns[np.argmax(correlations)]},
                  'mean': float(np.mean(correlations)),
                  'med': float(np.median(correlations)),
                  'std': float(np.std(correlations))}

    return [{'filename': curr_file,
             'type': type_file,
             'diff_columns': diff_columns,
             'correlation': corrValues,
             }]

def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    sessions = [args.in_sess1, args.in_sess2]

    # Get json file from task name
    dircmp = assert_fMRIPrep_structure(parser, [args.in_sess1, args.in_sess2])
    diff_files = dircmp.subdirs['func'].diff_files

    curr_json = []

    if diff_files:
        files_to_compare = filter_files(diff_files)
        for nKey in templates.keys():
            if files_to_compare[nKey]:
                curr_json = curr_json + compare_files(nKey,
                                                      files_to_compare[nKey],
                                                      sessions)

    elif dircmp.subdirs['func'].left_only + dircmp.subdirs['func'].right_only:
        logging.debug('Some files are only in some folder')
    else:
        logging.debug('This two fMRIPrep sessions ({}, {})'
                      ' are identical'.format(args.in_sess1, args.in_sess2))

    with open(args.out_file, 'w') as f:
        json.dump(curr_json, f, indent=4, separators=(',', ': '))



if __name__ == "__main__":
    main()
