import os, glob, sys
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='clean up, label, QC and bids-formats the triplets eye tracking dataset')
parser.add_argument('--in_path', type=str, required=True, help='absolute path to directory that contains all data (sourcedata)')
parser.add_argument('--run_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--out_path', type=str, default='./test.tsv', help='absolute path to output file')
args = parser.parse_args()

sys.path.append(os.path.join(args.run_dir, "pupil", "pupil_src", "shared_modules"))

from file_methods import PLData_Writer, load_pldata_file, load_object, save_object


def compile_file_list(in_path):

    col_names = ['subject', 'session', 'run', 'task', 'file_number', 'has_pupil', 'has_gaze', 'has_eyemovie', 'has_log']
    df_files = pd.DataFrame(columns=col_names)

    # on elm, for triplets : in_path = '/unf/eyetracker/neuromod/triplets/sourcedata'
    ses_list = sorted(glob.glob(f'{in_path}/sub-*/ses-*'))

    pupil_file_paths = []

    for ses_path in ses_list:
        [sub_num, ses_num] = ses_path.split('/')[-2:]
        events_list = sorted(glob.glob(f'{ses_path}/*task*events.tsv'))
        for event in events_list:
            ev_file = os.path.basename(event)
            [sub, ses, fnum, task_type, run_num, appendix] = ev_file.split('_')
            assert sub == sub_num
            assert ses_num == ses

            has_log = len(glob.glob(f'{ses_path}/{sub_num}_{ses_num}_{fnum}.log')) == 1
            pupil_path = f'{ses_path}/{sub_num}_{ses_num}_{fnum}.pupil'

            list_pupil = glob.glob(f'{pupil_path}/{task_type}_{run_num}/000/pupil.pldata')
            has_pupil = len(list_pupil) == 1
            if has_pupil:
                pupil_file_paths.append((os.path.dirname(list_pupil[0]), (sub, ses, run_num, task_type, fnum)))

            has_eyemv = len(glob.glob(f'{pupil_path}/{task_type}_{run_num}/000/eye0.mp4')) == 1
            has_gaze = len(glob.glob(f'{pupil_path}/{task_type}_{run_num}/000/gaze.pldata')) == 1

            run_data = [sub_num, ses_num, run_num, task_type, fnum, has_pupil, has_gaze, has_eyemv, has_log]
            #df_files = df_files.append(pd.Series(run_data, index=df_files.columns), ignore_index=True)
            df_files = pd.concat([df_files, pd.DataFrame(np.array(run_data).reshape(1, -1), columns=df_files.columns)], ignore_index=True)

    return df_files, pupil_file_paths


def export_and_plot(pupil_path, out_path):
    '''
    Function accomplishes two things:
    1. export gaze and pupil metrics from .pldata (pupil's) format to .npz format
    2. compile list of gaze and pupil positions (w timestamps and confidence), and export plots for visual QCing
    '''
    sub, ses, run, task, fnum = pupil_path[1]

    outpath_gaze = os.path.join(out_path, sub, ses)
    gfile_path = f'{outpath_gaze}/{sub}_{ses}_{run}_{fnum}_{task}_gaze2D.npz'

    if not os.path.exists(gfile_path):
        # note that gaze data includes pupil metrics from which each gaze was derived
        seri_gaze = load_pldata_file(pupil_path[0], 'gaze')[0]
        print(sub, ses, run, task, len(seri_gaze))

        # Convert serialized file to list of dictionaries...
        gaze_2plot_list = []
        deserialized_gaze = []

        for gaze in seri_gaze:
            gaze_data = {}
            gaze_2plot = np.empty(6) # [gaze_x, gaze_y, pupil_x, pupil_y, timestamp, confidence]
            for key in gaze.keys():
                if key != 'base_data': # gaze data
                    if key == 'norm_pos':
                        gaze_2plot[0: 2] = [gaze[key][0], gaze[key][1]]
                    elif key == 'timestamp':
                        gaze_2plot[4] = gaze[key]
                    elif key == 'confidence':
                        gaze_2plot[5] = gaze[key]
                    gaze_data[key] = gaze[key]
                else: # pupil data from which gaze was derived
                    gaze_pupil_data = {}
                    gaze_pupil = gaze[key][0]
                    for k in gaze_pupil.keys():
                        if k != 'ellipse':
                            if k == 'norm_pos':
                                gaze_2plot[2: 4] = [gaze_pupil[k][0], gaze_pupil[k][1]]
                            gaze_pupil_data[k] = gaze_pupil[k]
                        else:
                            gaze_pupil_ellipse_data = {}
                            for sk in gaze_pupil[k].keys():
                                gaze_pupil_ellipse_data[sk] = gaze_pupil[k][sk]
                            gaze_pupil_data[k] = gaze_pupil_ellipse_data
                    gaze_data[key] = gaze_pupil_data

            deserialized_gaze.append(gaze_data)
            gaze_2plot_list.append(gaze_2plot)

        print(len(deserialized_gaze))

        if len(deserialized_gaze) > 0:
            Path(outpath_gaze).mkdir(parents=True, exist_ok=True)
            np.savez(gfile_path, gaze2d = deserialized_gaze)

            # create and export QC plots per run
            array_2plot = np.stack(gaze_2plot_list, axis=0)

            fig, axes = plt.subplots(4, 1, figsize=(7, 14))
            plot_labels = ['gaze_x', 'gaze_y', 'pupil_x', 'pupil_x']

            for i in range(4):
                axes[i].scatter(array_2plot[:, 4]-array_2plot[:, 4][0], array_2plot[:, i], alpha=array_2plot[:, 5]*0.4)
                axes[i].set_ylim(-2, 2)
                axes[i].set_xlim(0, 350)
                axes[i].set_title(f'{sub} {task} {ses} {run} {plot_labels[i]}')

            outpath_fig = os.path.join(out_path, 'QC_gaze')
            Path(outpath_fig).mkdir(parents=True, exist_ok=True)

            fig.savefig(f'{outpath_fig}/{sub}_{ses}_{run}_{fnum}_{task}_QCplot.png')
            plt.close()


def main():
    # e.g., (elm): /unf/eyetracker/neuromod/triplets/sourcedata
    in_path = args.in_path
    out_path = args.out_path

    '''
    Step 1: compile overview of available files
    Export file list as .tsv
    '''
    file_report, pupil_paths = compile_file_list(in_path)

    outpath_report = os.path.join(out_path, 'QC_gaze')
    Path(outpath_report).mkdir(parents=True, exist_ok=True)
    file_report.to_csv(f'{outpath_report}/file_list.tsv', sep='\t', header=True, index=False)


    '''
    Step 2: export gaze files from pupil .pldata format to numpy .npz format
    For each run, plot the raw gaze & pupil data and export chart(s) for QCing
    '''
    for pupil_path in pupil_paths:
        export_and_plot(pupil_path, out_path)


    '''
    Step 3: manual QCing
    Rate the quality of each run based on graphs from step 2, log results in spreadsheet
    Compile a clean list of runs to drift-correct and bids-format
    Save run list as "QCed_file_list.tsv" in "out_path" directory
    Load this list to identify valid runs to processed with et_prep_step2.py
    '''

if __name__ == '__main__':
    sys.exit(main())
