#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Check txt data eprime with json file (BIDS)
    THEN
    Convert txt data from eprime to tsv extract_hcptrt
"""

import datetime
import glob
import logging
import numpy as np
import os

import bids
from convert_eprime.utils import remove_unicode

# tolerance for the match of scan_time to eprime file
TIME_CHECK_DELTA_TOL = 240


def get_diff_time(scan_time, eprime_time):
    scan_time_local = datetime.datetime.strptime(scan_time, '%H:%M:%S.%f')
    scan_time_local = scan_time_local.replace(microsecond=0)
    eprime_time_local = datetime.datetime.strptime(eprime_time[0],
                                                   '%H:%M:%S') + datetime.timedelta(milliseconds=eprime_time[1])

    diff_time = scan_time_local - eprime_time_local

    return int(diff_time.total_seconds()), eprime_time_local


def get_closest_eprime(eprime_files, scan_time):

    min_diff = np.Inf
    eprime_choosen = ''
    all_vals = []
    for eprime_file in eprime_files:
        # Read task_file_path
        with open(eprime_file, 'rb') as fo:
            text_data = list(fo)

        # Remove unicode characters.
        filtered_data = [remove_unicode(row.decode('utf-8', 'ignore')) for row in text_data]
        res = [i for i in filtered_data if 'SessionTime' in i]
        start_scan = [i for i in filtered_data if "SyncSlide.OnsetTime" in i or
                                                  "GetReady.FinishTime" in i or
                                                  "CountDownSlide" in i]

        if len(start_scan) == 0:
            start_scan = 0
            logging.debug('File with  {}'.format(eprime_file))
        else:
            start_scan = int(start_scan[0].split()[-1])

        eprime_time = res[0].split()[-1]

        diff_scan_eprime, eprime_time_local = get_diff_time(scan_time, (eprime_time, start_scan))
        all_vals.append((eprime_file, diff_scan_eprime))

        if np.abs(diff_scan_eprime) < min_diff:
            min_diff = np.abs(diff_scan_eprime)
            eprime_choosen = eprime_file

    return min_diff, eprime_choosen, all_vals


def main():
    logging.basicConfig(level=logging.DEBUG)
    layout = bids.BIDSLayout('/home/bore/p/neuromod/data/hcptrt/')
    non_rest_tasks = [t for t in layout.get_tasks() if t != 'restingstate']
    task_bolds = layout.get(suffix='bold', extension='.nii.gz', task=non_rest_tasks)

    eprime_path = os.path.join(layout.root, 'sourcedata')

    for task_bold in task_bolds:
        logging.debug(task_bold.filename)
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
            logging.debug('Found {} with {} with diff {}s'.format(task_bold.filename,
                                                                  eprime_file,
                                                                  min_diff))
            out_tsv_path = task_bold.path.replace('_bold.nii.gz', '_event.tsv')

        elif min_diff == np.Inf:
            print('ERROR')
            print('Candidates: {}'.format(all_vals))
            print(eprime_files)
            print(eprime_files_research)
        elif min_diff>0:
            logging.debug('Eprime was started before MRI')
            logging.debug('Found {} with {} with diff {}s'.format(task_bold.filename,
                                                                  eprime_file,
                                                                  min_diff))
        elif min_diff<0:
            logging.debug('Eprime was started AFTER MRI')
            logging.debug('ERRROR -> No eprime were found with {} {}'.format(task_bold.filename, scan_time))
            logging.debug('ERRROR -> Candidates: {}'.format(all_vals))
            logging.debug('min_diff choosen: {}'.format(min_diff))

        logging.debug('---------------------------------------------------')
        # convert_event_file(eprime_file, ents['task'], out_tsv_path)


if __name__ == "__main__":
    main()
