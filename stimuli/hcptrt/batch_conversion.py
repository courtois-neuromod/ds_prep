#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Convert txt data from eprime to tsv extract_hcptrt
"""

import argparse
import datetime
import glob
import logging
import numpy as np
import os

import bids

from extract_hcptrt import convert_event_file

# tolerance for the match of scan_time to eprime file
TIME_CHECK_DELTA_TOL = 240


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_BIDS',
                   help='BIDS structure with HCPTRT tasks.')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    p.add_argument('-v', action='store_true', dest='verbose',
                   help='If set, produces verbose output.')
    return p


def get_diff_time(scan_time, eprime_time):
    scan_time = datetime.datetime.strptime(scan_time, '%H:%M:%S.%f')
    diff_time = scan_time - eprime_time

    return int(diff_time.total_seconds()), eprime_time


def get_closest_eprime(eprime_files, scan_time):

    min_diff = np.Inf
    eprime_choosen = ''
    all_vals = []
    for eprime_file in eprime_files:
        # Dummy values to make it crash if
        session_time = 0
        onset_time = '0'

        # Read task_file_path
        with open(eprime_file, 'r', encoding='utf-16-le') as fo:
            initial_tr_marker_found = False

            for line in fo:
                if 'SessionTime' in line:
                    session_time = line.split(' ')[1].strip()
                    session_time = datetime.datetime.strptime(session_time, '%H:%M:%S')
                if 'InitialTR' in line or 'CountDownPROC' in line:
                    initial_tr_marker_found = True
                if ('OnsetTime' in line and initial_tr_marker_found) or ('GetReady.FinishTime' in line):
                    onset_time = int(line.split(' ')[1].strip())
                    onset_time = datetime.timedelta(seconds=onset_time/1000.)
                    break

        # first ttl
        eprime_time = session_time + onset_time

        diff_scan_eprime, eprime_time_local = get_diff_time(scan_time, eprime_time)
        all_vals.append((eprime_file, diff_scan_eprime))

        if np.abs(diff_scan_eprime) < min_diff:
            min_diff = np.abs(diff_scan_eprime)
            eprime_choosen = eprime_file

    return min_diff, eprime_choosen, all_vals


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    verbose = args.verbose
    overwrite = args.overwrite
    in_BIDS = args.in_BIDS

    if verbose:
        logging.basicConfig(level=logging.INFO)

    layout = bids.BIDSLayout(in_BIDS)
    non_rest_tasks = [t for t in layout.get_tasks() if t != 'restingstate']
    task_bolds = layout.get(suffix='bold', extension='.nii.gz', task=non_rest_tasks)

    eprime_path = os.path.join(layout.root, 'sourcedata')

    for task_bold in task_bolds:
        logging.info('Input: {}'.format(task_bold.filename))
        scan_time = task_bold.get_metadata()['AcquisitionTime']
        ents = task_bold.entities

        eprime_files_research = os.path.join(eprime_path,
                                             'sub-%s'%ents['subject'],
                                             'ses-%s'%ents['session'],
                                             'func',
                                             'p%02d_%s*.txt'%(int(ents['subject']),
                                                              ents['task'].upper()))
        eprime_files = glob.glob(eprime_files_research)

        eprime_files = [i for i in eprime_files if 'runp' not in i and
                                                   'runc' not in i]

        min_diff, eprime_file, all_vals = get_closest_eprime(eprime_files, scan_time)

        if np.abs(min_diff) < TIME_CHECK_DELTA_TOL:
            logging.info('Found {} with {} with diff {}s'.format(task_bold.filename,
                                                                  eprime_file,
                                                                  min_diff))
            out_tsv_path = task_bold.path.replace('_bold.nii.gz', '_events.tsv')
            convert_event_file(eprime_file,
                               ents['task'],
                               out_tsv_path,
                               verbose=verbose,
                               overwrite=overwrite)
        if min_diff == np.Inf:
            logging.info('ERROR')
            logging.info('Candidates: {}'.format(all_vals))
            logging.info(eprime_files)
            logging.info(eprime_files_research)
        elif min_diff > 0 and np.abs(min_diff) >= TIME_CHECK_DELTA_TOL:
            logging.info('WARNING - Eprime was started before MRI')
            logging.info('Found {} with {} with diff {}s'.format(task_bold.filename,
                                                                  eprime_file,
                                                                  min_diff))
        elif min_diff < 0:
            logging.info('Eprime was started AFTER MRI')
            logging.info('ERRROR -> No eprime were found with {} {}'.format(task_bold.filename, scan_time))
            logging.info('ERRROR -> Candidates: {}'.format(all_vals))
            logging.info('min_diff choosen: {}'.format(min_diff))

        logging.info('---------------------------------------------------')


if __name__ == "__main__":
    main()
