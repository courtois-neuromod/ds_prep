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


def make_gaze_figs(df_allruns, out_path):

    cols_to_print = ['Below 0.85 Confidence Threshold', 'Outside Screen Area', 'X diff from mid',
                     'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept']

    chart_titles = ['Percent gaze below confidence threshold', 'Percent gaze outside screen area',
                    'Average normalized squared distance from middle in X',
                    'Average normalized squared distance from middle in Y',
                    'Slope of gaze position in X over time',
                    'Intercept of gaze position in X',
                    'Slope of gaze position in Y over time',
                    'Intercept of gaze position in Y']

    yaxis_titles = ['Percent gaze', 'Percent gaze',
                    'Squared distance in X',
                    'Squared distance in Y',
                    'Slope in X',
                    'Intercept in X',
                    'Slope in Y',
                    'Intercept in Y']

    file_names = ['subtreshold', 'outarea', 'XfromMid', 'YfromMid', 'Xslope', 'Xintercept', 'Yslope', 'Yintercept']

    df_allruns_nocalib = df_allruns[df_allruns['Type']=='Run']
    #df_allruns_calib = df_allruns[df_allruns['Type']=='Calib']

    x_val = df_allruns_nocalib['Sub'].to_numpy().astype('int')
    x_val[x_val > 5] = 4

    x_subs = [1, 2, 3, 4]
    x_labels = ['Sub-01', 'Sub-02', 'Sub-03', 'Sub-06']

    for i in range(len(cols_to_print)):
        plt.clf()
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        y_val = df_allruns_nocalib[cols_to_print[i]].to_numpy()
        plt.scatter(x_val, y_val, s = 100, alpha=0.2)
        plt.xticks(x_subs, x_labels, rotation='horizontal')

        plt.xlabel('Participants', labelpad=20)
        plt.ylabel(yaxis_titles[i], labelpad=20)
        plt.title(chart_titles[i])

        plt.savefig(os.path.join(out_path, file_names[i] + '.png'))



def crunch_data(list_gazefiles, list_pupilfiles, out_path):
    '''
    For each subject, performance metrics are extracted per run, per session (scan) and per run
    and overall (excluding session 1, which differs from the subsequent ones)

    Input:
        list_files: sorted list of str that correspond to session's .tsv file of gaze report
        out_path: str path to output directory
    Output:
        None : exports .tsv files of computed scores saved in specified output directory (default is '/results')
    '''
    col_names = ['Name', 'Type', 'Processing', 'Sub', 'Sess', 'Run', 'Below 0.85 Confidence Threshold',
                 'Outside Screen Area', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept',
                 'Y slope', 'Y intercept']

    df_allruns = pd.DataFrame(columns=col_names)

    for gazefile_path in list_gazefiles:

        split_file = os.path.basename(gazefile_path).split('_')
        sub_id = int(split_file[0][-1])
        ses_id = split_file[1][-3:]

        gaze_file = pd.read_csv(gazefile_path, sep = '\t')
        gaze_file.insert(loc=3, column='Sub', value=sub_id, allow_duplicates=True)
        gaze_file.insert(loc=4, column='Sess', value=ses_id, allow_duplicates=True)

        df_allruns = pd.concat((df_allruns, gaze_file), ignore_index=True)

    df_allruns.to_csv(os.path.join(out_path, 'Allsessions_gaze_report.tsv'), sep='\t', header=True, index=False)

    make_gaze_figs(df_allruns, out_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--idir', type=str, required=True, help='path to bids folder of .tsv behavioural output files')
    parser.add_argument('-o', '--odir', type=str, default='./results', help='path to output directory')
    args = parser.parse_args()

    in_path = args.idir
    out_path = args.odir

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    list_gazefiles = sorted(glob.glob(os.path.join(in_path, 's0*', 'ses-0*', 'qc', 'sub-*_gaze_report.tsv')))
    list_pupilfiles = sorted(glob.glob(os.path.join(in_path, 's0*', 'ses-0*', 'qc', 'sub-*_gaze_report.tsv')))

    crunch_data(list_gazefiles, list_pupilfiles, out_path)


if __name__ == '__main__':
    sys.exit(main())
