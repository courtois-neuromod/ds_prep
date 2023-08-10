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


# Assign run number to subtasks for retino and fLoc
# https://github.com/courtois-neuromod/task_stimuli/blob/main/src/sessions/ses-retino.py
# https://github.com/courtois-neuromod/task_stimuli/blob/main/src/sessions/ses-floc.py
run2task_mapping = {
    'retino': {
        'task-wedges': 'run-01',
        'task-rings': 'run-02',
        'task-bar': 'run-03'
    },
    'floc': {
        'task-flocdef': 'run-01',
        'task-flocalt': 'run-02'
    }
}

def compile_file_list(in_path):

    col_names = ['subject', 'session', 'run', 'task', 'file_number', 'has_pupil', 'has_gaze', 'has_eyemovie', 'has_log', 'empty_log']
    df_files = pd.DataFrame(columns=col_names)

    task_root = in_path.split('/')[-2]

    # on elm, for triplets : in_path = '/unf/eyetracker/neuromod/triplets/sourcedata'
    ses_list = sorted(glob.glob(f'{in_path}/sub-*/ses-0*'))

    pupil_file_paths = []

    for ses_path in ses_list:
        [sub_num, ses_num] = ses_path.split('/')[-2:]

        if task_root == 'friends':
            epfile_list = sorted(glob.glob(f'{ses_path}/sub-*.pupil/task-friends-*/000'))

            for epfile in epfile_list:
                f1, f2 = epfile.split('/')[-3:-1]
                t1, t2, run_num = f2.split('-')
                task_type = f'{t1}-{t2}'
                sub, ses, fnum = f1.split('.')[0].split('_')

                if sub in ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']:
                    assert sub == sub_num
                    assert ses_num == ses

                    log_list = glob.glob(f'{ses_path}/{sub_num}_{ses_num}_{fnum}.log')
                    has_log = len(log_list) == 1
                    if has_log:
                        with open(log_list[0]) as f:
                            lines = f.readlines()
                            empty_log = len(lines) == 0
                    else:
                        empty_log = True

                    list_pupil = glob.glob(f'{epfile}/pupil.pldata')
                    has_pupil = len(list_pupil) == 1
                    if has_pupil:
                        pupil_file_paths.append((os.path.dirname(list_pupil[0]), (sub, ses, run_num, task_type, fnum)))

                    has_eyemv = len(glob.glob(f'{epfile}/eye0.mp4')) == 1
                    has_gaze = len(glob.glob(f'{epfile}/gaze.pldata')) == 1

                    run_data = [sub_num, ses_num, run_num, task_type, fnum, has_pupil, has_gaze, has_eyemv, has_log, empty_log]
                    df_files = pd.concat([df_files, pd.DataFrame(np.array(run_data).reshape(1, -1), columns=df_files.columns)], ignore_index=True)

        else:
            events_list = sorted(glob.glob(f'{ses_path}/*task*events.tsv'))
            for event in events_list:
                ev_file = os.path.basename(event)

                try:
                    if task_root in ['retino', 'floc']:
                        skip_run_num = True
                        [sub, ses, fnum, task_type, appendix] = ev_file.split('_')
                        run_num = run2task_mapping[task_root][task_type]
                    else:
                        skip_run_num = False
                        [sub, ses, fnum, task_type, run_num, appendix] = ev_file.split('_')

                    if sub in ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']:
                        assert sub == sub_num
                        assert ses_num == ses

                        log_list = glob.glob(f'{ses_path}/{sub_num}_{ses_num}_{fnum}.log')
                        has_log = len(log_list) == 1
                        if has_log:
                            with open(log_list[0]) as f:
                                lines = f.readlines()
                                empty_log = len(lines) == 0
                        else:
                            empty_log = True

                        if skip_run_num:
                            pupil_path = f'{ses_path}/{sub_num}_{ses_num}_{fnum}.pupil/{task_type}/000'
                        else:
                            pupil_path = f'{ses_path}/{sub_num}_{ses_num}_{fnum}.pupil/{task_type}_{run_num}/000'

                        list_pupil = glob.glob(f'{pupil_path}/pupil.pldata')
                        has_pupil = len(list_pupil) == 1
                        if has_pupil:
                            pupil_file_paths.append((os.path.dirname(list_pupil[0]), (sub, ses, run_num, task_type, fnum)))

                        has_eyemv = len(glob.glob(f'{pupil_path}/eye0.mp4')) == 1
                        has_gaze = len(glob.glob(f'{pupil_path}/gaze.pldata')) == 1

                        run_data = [sub_num, ses_num, run_num, task_type, fnum, has_pupil, has_gaze, has_eyemv, has_log, empty_log]
                        df_files = pd.concat([df_files, pd.DataFrame(np.array(run_data).reshape(1, -1), columns=df_files.columns)], ignore_index=True)
                except:
                    print(f'cannot process {ev_file}')

    return df_files, pupil_file_paths


def export_and_plot(pupil_path, in_path, out_path):
    '''
    Function accomplishes two things:
    1. export gaze and pupil metrics from .pldata (pupil's) format to .npz format
    2. compile list of gaze and pupil positions (w timestamps and confidence), and export plots for visual QCing
    '''
    sub, ses, run, task, fnum = pupil_path[1]

    task_root = out_path.split('/')[-1]
    pseudo_task = 'task-mario3' if task_root == 'mario3' else task

    outpath_gaze = os.path.join(out_path, sub, ses)
    gfile_path = f'{outpath_gaze}/{sub}_{ses}_{run}_{fnum}_{pseudo_task}_gaze2D.npz'

    if not os.path.exists(gfile_path):
        # note that gaze data includes pupil metrics from which each gaze was derived
        seri_gaze = load_pldata_file(pupil_path[0], 'gaze')[0]
        print(sub, ses, run, pseudo_task, len(seri_gaze))

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

        try:
            if task_root == 'friends':
                run_dur = 770
            else:
                if task_root in ['retino', 'floc']:
                    ev_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}_{task}_events.tsv'
                else:
                    ev_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}_{task}_{run}_events.tsv'
                ev_lasttrial = pd.read_csv(ev_path, sep='\t', header=0).iloc[-1]
                run_dur = int(ev_lasttrial['onset'] + 20)
        except:
            print('event file did not load, using default run duration')
            run_dur = 700

        if len(deserialized_gaze) > 0:
            Path(outpath_gaze).mkdir(parents=True, exist_ok=True)
            np.savez(gfile_path, gaze2d = deserialized_gaze)

            # create and export QC plots per run
            array_2plot = np.stack(gaze_2plot_list, axis=0)

            fig, axes = plt.subplots(4, 1, figsize=(7, 14))
            plot_labels = ['gaze_x', 'gaze_y', 'pupil_x', 'pupil_x']

            for i in range(4):
                axes[i].scatter(array_2plot[:, 4]-array_2plot[:, 4][0], array_2plot[:, i], c=array_2plot[:, 5], s=10, cmap='terrain_r', alpha=0.2)#alpha=array_2plot[:, 5]*0.4)
                axes[i].set_ylim(-2, 2)
                axes[i].set_xlim(0, run_dur)
                axes[i].set_title(f'{sub} {pseudo_task} {ses} {run} {plot_labels[i]}')

            outpath_fig = os.path.join(out_path, 'QC_gaze')
            Path(outpath_fig).mkdir(parents=True, exist_ok=True)

            fig.savefig(f'{outpath_fig}/{sub}_{ses}_{run}_{fnum}_{pseudo_task}_QCplot.png')
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
        export_and_plot(pupil_path, in_path, out_path)


    '''
    Step 3: manual QCing
    Rate the quality of each run based on graphs from step 2, log results in spreadsheet
    Compile a clean list of runs to drift-correct and bids-format
    Save run list as "QCed_file_list.tsv" in "out_path" directory
    Load this list to identify valid runs to processed with et_prep_step2.py
    '''

if __name__ == '__main__':
    sys.exit(main())
