#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Check txt data eprime with json file (BIDS)
    THEN
    Convert txt data from eprime to tsv extract_hcptrt
"""

import datetime
import logging
import numpy as np
import pathlib
import tz

import bids
from convert_eprime.utils import remove_unicode
from extract_hcptrt import convert_event_file

# tolerance for the match of scan_time to eprime file
TIME_CHECK_DELTA_TOL = 180


def assert_valide_combination(scan_time, eprime_time, in_filenames):

    scan_time_local = datetime.datetime.strptime(scan_time, '%H:%M:%S.%f')
    scan_time_local = scan_time_local.replace(microsecond=0)
    eprime_time_utc = datetime.datetime.strptime(eprime_time, '%H:%M:%S')

    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    # Tell the datetime object that it's in UTC time zone since
    # datetime objects are 'naive' by default
    eprime_time_utc = eprime_time_utc.replace(tzinfo=from_zone)

    # Convert time zone
    eprime_time_local = eprime_time_utc.astimezone(to_zone)

    if np.abs(scan_time_local - eprime_time_local) < TIME_CHECK_DELTA_TOL:
        return True
    else:
        logging.info('You try to combine {} and {} but do not have the '
                     'same time stamp ({} and {}).'.format(in_filenames[0],
                                                           in_filenames[1],
                                                           scan_time_local.strftime("%H:%M:%S"),
                                                           eprime_time_local.strftime("%H:%M:%S")))
        return False


def main():
    layout = bids.BIDSLayout('./hcptrt')
    non_rest_tasks = [t for t in layout.get_tasks() if t != 'restingstate']
    task_bolds = layout.get(suffix='bold', extension='.nii.gz', task=non_rest_tasks)

    eprime_path = layout.root / 'sourcedata' / 'eprime'

    for task_bold in task_bolds:
        scan_time = task_bold.get_metadata()['AcquisitionTime']
        ents = task_bold.entities
        task_file_path = eprime_path / \
            'sub-%s'%ents['subject'] / \
            'ses-%s'%ents['session'] / \
            'p%02d_%s.txt'%(ents['subject'], ents['task'].upper())

        # Read task_file_path
        with open(task_file_path, 'rb') as fo:
            text_data = list(fo)

        # Remove unicode characters.
        filtered_data = [remove_unicode(row.decode('utf-8', 'ignore')) for row in text_data]
        res = [i for i in filtered_data if 'SessionStartDateTimeUtc' in i]
        eprime_time = res[0].split()[-1]
        paths_bold_eprime = [task_bold.filename,
                             pathlib.Path(task_file_path).name]

        # Check DELTA
        assert_valide_combination(scan_time, eprime_time, paths_bold_eprime)

        out_tsv_path = task_bold.path.replace('_bold.nii.gz', '_event.tsv')
        convert_event_file(task_file_path, ents['task'], out_tsv_path)
