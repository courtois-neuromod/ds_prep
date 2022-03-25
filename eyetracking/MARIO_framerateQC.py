import os, glob
import sys

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

'''
Function parses through session's .log file and plots
    1. a distribution of intervals between 60 frames of video
    2. the difference between a frame's logged time and its target time, based on fps rate (from .bk2 onset)
'''

def get_arguments():

    parser = argparse.ArgumentParser(description='Assesses skips between video frames')
    parser.add_argument('--file_path', default='', type=str, help='absolute path to session directory that contains log file(s)')
    parser.add_argument('--out_path', default='', type=str, help='absolute path to output directory')
    args = parser.parse_args()

    return args


def qc_videoskips(logfile_path, out_path):

    step_times = []
    bk2_list = []

    try:
        with open(logfile_path, 'r') as log:
            for line in log:

                timestamp, entry_type, message = line.split('\t')
                if 'VideoGame: recording movie in' in message:
                    bk2_list.append(os.path.basename(message.split(' ')[-1][:-1]).split('.')[0])
                    step_times.append([])
                elif 'level step:' in message:
                    step_times[-1].append(float(timestamp))

    except:
        print(os.path.basename(logfile_path) + " has no output.")

    if len(step_times) > 0:
        # Figure plots intervals
        log_name = os.path.basename(logfile_path).split('.')[0]
        concat_intervals = []

        for steps in step_times:
            reset_steps = (np.array(steps) - steps[0]).tolist()
            intervals = []
            for i in range(len(reset_steps) - 1):
                interval = reset_steps[i+1] - reset_steps[i]
                intervals.append(interval)
            concat_intervals += intervals

        plt.hist(concat_intervals, bins=100)
        plt.xlabel('Distribution of intervals between sets of 60 frames as fps=60, in s')
        plt.ylabel('Frequency')

        plt.title(log_name)
        plt.savefig(os.path.join(out_path, log_name + '_intervals.png'))

        # Figure of lag from target time (.bk2 logged onset is 0)
        plt.clf()

        for i in range(len(step_times)):
            step = step_times[i]
            bkname = bk2_list[i]
            reset_steps = np.array(step) - step[0]
            target_steps = np.array(range(len(step)))
            off_time = reset_steps - target_steps
            plt.plot(off_time, label=bkname)

        plt.xlabel('.bk2 time since onset, in s')
        plt.ylabel('Lag from target time, in s')

        plt.title(log_name)
        plt.savefig(os.path.join(out_path, log_name + '_lags.png'))


if __name__ == '__main__':
    args = get_arguments()

    file_path = args.file_path
    out_path = args.out_path

    log_list = glob.glob(os.path.join(file_path, '*.log'))

    for logfile in log_list:
        qc_videoskips(logfile, out_path)
