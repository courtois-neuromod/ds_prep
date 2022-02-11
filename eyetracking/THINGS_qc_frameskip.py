import os
import sys

import argparse
import glob

import numpy as np
from numpy import nan as NaN
import pandas as pd
#from scipy.stats import norm
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt



def crunch_ff(list_calib_ff, list_run_ff):

    c_count = 0
    r_count = 0
    s1_count = s2_count = s3_count = s6_count = 0
    s1_ccount = s2_ccount = s3_ccount = s6_ccount = 0

    for calib_ff in list_calib_ff:

        run_num = os.path.basename(calib_ff).split('_')[0]
        sub_id, ses_id, _, _ = calib_ff.split('/')[-4:]

        calib_file = pd.read_csv(calib_ff, header=None, sep = '\t')
        maxgaptime = np.max(calib_file.iloc[:, 2].to_numpy())

        if maxgaptime > 3.0: #0.0125: # in seconds
            c_count += 1
            if sub_id == 's01':
                s1_ccount +=1
            elif sub_id == 's02':
                s2_ccount +=1
            elif sub_id =='s03':
                s3_ccount += 1
            elif sub_id =='s06':
                s6_ccount += 1
            print('CALIB', sub_id, ses_id, run_num, str(maxgaptime))

    for run_ff in list_run_ff:

        run_num = os.path.basename(run_ff).split('_')[0]
        sub_id, ses_id, _, _ = run_ff.split('/')[-4:]

        run_file = pd.read_csv(run_ff, header=None, sep = '\t')
        maxgaptime = np.max(run_file.iloc[:, 2].to_numpy())

        if maxgaptime > 3.0: #0.0125: # in seconds
            r_count += 1
            if sub_id == 's01':
                s1_count +=1
            elif sub_id == 's02':
                s2_count +=1
            elif sub_id =='s03':
                s3_count += 1
            elif sub_id =='s06':
                s6_count += 1
            print('RUN', sub_id, ses_id, run_num, str(maxgaptime))

    print(c_count, r_count)
    print(s1_ccount, s2_ccount, s3_ccount, s6_ccount)
    print(s1_count, s2_count, s3_count, s6_count)


def main():
    '''
    Quick and dirty QC script for the THINGS dataset that
    can be ran after THINGS_qualitycheck_summary.py

    Crunches up additional stats on skipped frames (eye camera) based on
    run*_calib_framegaps.tsv and run*_run_framegaps.tsv files outputed by quality_check_THINGS_summary.py

    Prints tally of the number of runs with at least one skip above a certain lenght (in s),
    per subject and overall. The duraction of the gap threshold is specified on lines 32 and 52

    No output
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--idir', type=str, default='/home/labopb/Documents/Marie/neuromod/THINGS/Eye-tracking/offline_calibration')
    args = parser.parse_args()

    in_path = args.idir

    list_calib_ff = sorted(glob.glob(os.path.join(in_path, 's0*', 'ses-0*', 'qc', 'run*_calib_framegaps.tsv')))
    list_run_ff = sorted(glob.glob(os.path.join(in_path, 's0*', 'ses-0*', 'qc', 'run*_run_framegaps.tsv')))

    print(len(list_calib_ff), len(list_run_ff))

    crunch_ff(list_calib_ff, list_run_ff)


if __name__ == '__main__':
    sys.exit(main())
