#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Check txt data eprime with json file (BIDS)
    THEN
    Convert txt data from eprime to tsv extract_hcptrt
"""

import datetime
from dateutil import tz
import glob
import logging
import numpy as np
import os
import pathlib

import bids
from convert_eprime.utils import remove_unicode
from extract_hcptrt import convert_event_file

# tolerance for the match of scan_time to eprime file
TIME_CHECK_DELTA_TOL = 240


def get_diff_time(scan_time, eprime_time):
    scan_time_local = datetime.datetime.strptime(scan_time, '%H:%M:%S.%f')
    scan_time_local = scan_time_local.replace(microsecond=0)
    #eprime_time_utc = datetime.datetime.strptime(eprime_time, '%H:%M:%S')
    eprime_time_local = datetime.datetime.strptime(eprime_time, '%H:%M:%S')
    #from_zone = tz.tzutc()
    #to_zone = tz.tzlocal()

    # Tell the datetime object that it's in UTC time zone since
    # datetime objects are 'naive' by default
    #scan_time_local = scan_time_local.replace(tzinfo=to_zone)
    #eprime_time_utc = eprime_time_utc.replace(tzinfo=from_zone)

    # Convert time zone
    #eprime_time_local = eprime_time_utc.astimezone(to_zone)

    diff_time = scan_time_local - eprime_time_local

    return np.abs(int(diff_time.total_seconds())), eprime_time_local


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
        eprime_time = res[0].split()[-1]

        diff_scan_eprime, eprime_time_local = get_diff_time(scan_time, eprime_time)
        all_vals.append((eprime_file, diff_scan_eprime))

        if diff_scan_eprime < min_diff:
            min_diff = diff_scan_eprime
            eprime_choosen = eprime_file

    return min_diff, eprime_choosen, all_vals


def main():
    logging.basicConfig(level=logging.DEBUG)
    layout = bids.BIDSLayout('/home/bore/p/neuromod/data/hcptrt/')
    non_rest_tasks = [t for t in layout.get_tasks() if t != 'restingstate']
    task_bolds = layout.get(suffix='bold', extension='.nii.gz', task=non_rest_tasks, subject='01')

    eprime_path = os.path.join(layout.root, 'sourcedata')

    for task_bold in task_bolds:
        print(task_bold.filename)
        scan_time = task_bold.get_metadata()['AcquisitionTime']
        ents = task_bold.entities

        print('Subject: {}'.format(ents['subject']))
        print('Task: {}'.format(ents['task']))
        print('Session: {}'.format(ents['session']))


        eprime_files = glob.glob(os.path.join(eprime_path,
                                      'sub-%s'%ents['subject'],
                                      'ses-%s'%ents['session'],
                                      'func',
                                      'p%02d_%s*.txt'%(int(ents['subject']),
                                                           ents['task'].upper())))


        min_diff, eprime_file, all_vals = get_closest_eprime(eprime_files, scan_time)

        if min_diff < TIME_CHECK_DELTA_TOL:
            print('Found {} with {} with diff {}s'.format(task_bold.filename,
                                            eprime_file,
                                            min_diff))
            out_tsv_path = task_bold.path.replace('_bold.nii.gz', '_event.tsv')
        else:
            print('ERRROR -> No eprime were found with {} {}'.format(task_bold.filename, scan_time))
            print('ERRROR -> Candidates: {}'.format(all_vals))
        #convert_event_file(eprime_file, ents['task'], out_tsv_path)

if __name__ == "__main__":
    main()
