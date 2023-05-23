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
parser.add_argument('--phase', type=str, required=True, choices=['A', 'B'])
parser.add_argument('--cthresh', default=0.75, type=float, help='confidence threshold for high quality gaze to use for drift correction')
parser.add_argument('--run_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--out_path', type=str, default='./test.tsv', help='absolute path to output file')
args = parser.parse_args()

#args.run_dir = /home/mariestl/cneuromod/ds_prep/eyetracking
sys.path.append(os.path.join(args.run_dir, "pupil", "pupil_src", "shared_modules"))
#sys.path.append(os.path.join("/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking", "pupil", "pupil_src", "shared_modules"))

#from video_capture.file_backend import File_Source
from file_methods import PLData_Writer, load_pldata_file, load_object, save_object
#from gaze_producer.worker.fake_gpool import FakeGPool, FakeIPC

#from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
#from gaze_mapping.gazer_2d import Gazer2D
#from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
#from gaze_mapping.gazer_3d.gazer_headset import Gazer3D


# List CNeuromod1 datasets that include eyetracking data
'''
ds_specs = {
    'emotionsvideos': {},
    'floc': {},
    'friends': {},
    'friends_fix': {},
    'harrypotter': {},
    'mario': {},
    'mario3': {},
    'mariostars': {},
    'movie10fix': {},
    'narratives': {},
    'petitprince': {},
    'retino': {},
    'shinobi': {},
    'things': {},
    'triplets': {}
}
'''


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


def create_gaze_path(row, file_path):
    '''
    for each run, create path to deserialized gaze file
    '''
    s = row['subject']
    ses = row['session']
    return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["run"]}_{row["file_number"]}_{row["task"]}_gaze2D.npz'


def create_event_path(row, file_path, log=False):
    '''
    for each run, create path to events.tsv file
    '''
    s = row['subject']
    ses =row['session']
    if log:
        return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["file_number"]}.log'
    else:
        return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["file_number"]}_{row["task"]}_{row["run"]}_events.tsv'


def get_onset_time(log_path, run_num, task):
    onset_time_dict = {}
    rnum = 'run-00'

    if task == 'task-wordsfamiliarity':
        with open(log_path) as f:
            lines = f.readlines()
            for line in lines:
                if "Imported data/language/triplets/words_designs" in line:
                    rnum = line.split('\t')[-1].split(' ')[1].split('/')[-1].split('_')[-2]
                elif "fMRI TTL 0" in line:
                    onset_time = line.split('\t')[0]
                    onset_time_dict[rnum] = onset_time

    elif task == 'task-triplets':
        with open(log_path) as f:
            lines = f.readlines()
            for line in lines:
                if "Imported data/language/triplets/designs" in line:
                    rnum = line.split('\t')[-1].split(' ')[1].split('/')[-1].split('_')[-2]
                elif "fMRI TTL 0" in line:
                    onset_time = line.split('\t')[0]
                    onset_time_dict[rnum] = onset_time

    return float(onset_time_dict[run_num])


def reset_gaze_time(gaze, onset_time, conf_thresh=0.9):
    '''
    Realign gaze timestamps based on the task & eyetracker onset (triggered by fMRI run start)
    Export new list of gaze dictionaries (w task-aligned time stamps) and other metrics needed to perform drift correction
    '''
    # all gaze values (unfiltered)
    reset_gaze_list = []
    all_x = []
    all_y = []
    all_times = []
    all_conf = []

    # normalized distance (proportion of screen) between above-threshold gaze and fixation point
    # includes fixation and trial gaze
    clean_dist_x = []
    clean_dist_y = []
    clean_times = []
    clean_conf = [] # probably not needed since filtered...

    for gz_pt in gaze:
        timestp = gz_pt['timestamp'] - onset_time
        # exclude gaze & pupil gathered before task onset
        if timestp > 0.0:
            x_norm, y_norm = gz_pt['norm_pos']
            cfd = gz_pt['confidence']

            all_x.append(x_norm)
            all_y.append(y_norm)
            all_times.append(timestp)
            all_conf.append(cfd)

            gz_pt['reset_time'] = timestp
            reset_gaze_list.append(gz_pt)

            if cfd > conf_thresh:
                clean_dist_x.append(x_norm - 0.5)
                clean_dist_y.append(y_norm - 0.5)
                clean_conf.append(cfd)
                clean_times.append(timestp)

    return reset_gaze_list, (all_x, all_y, all_times, all_conf), (clean_dist_x, clean_dist_y, clean_times, clean_conf)


def get_fixation_gaze(df_ev, clean_dist_x, clean_dist_y, clean_times, clean_conf):
    '''
    Identify gaze that correspond to periods of fixation
    '''
    fix_dist_x = []
    fix_dist_y = []
    fix_times = []
    fix_conf = []

    j = 0

    for i in range(df_ev.shape[0]):
        trial_onset = df_ev['onset'][i]
        trial_offset = trial_onset + df_ev['duration'][i]
        fix_offset = trial_offset + df_ev['isi'][i]
        if i == 0:
            # add gaze from very first fixation at run onset (before first trial)
            while j < len(clean_times) and clean_times[j] < (trial_onset - 0.1):
                if clean_times[j] > 3.0: # cut off first few seconds for cleaner fixations
                    fix_dist_x.append(clean_dist_x[j])
                    fix_dist_y.append(clean_dist_y[j])
                    fix_times.append(clean_times[j])
                    fix_conf.append(clean_conf[j])
                j += 1

        while j < len(clean_times) and clean_times[j] < fix_offset:
            # + 0.8 = 800ms (0.8s) after trial offset to account for saccade
            if clean_times[j] > (trial_offset + 0.8) and clean_times[j] < (fix_offset - 0.1):
                fix_dist_x.append(clean_dist_x[j])
                fix_dist_y.append(clean_dist_y[j])
                fix_times.append(clean_times[j])
                fix_conf.append(clean_conf[j])
            j += 1

    return fix_dist_x, fix_dist_y, fix_times, fix_conf


def assign_gazeConf2trial(df_ev, vals_times, vals_conf, conf_thresh=0.9):

    gazeconf_per_trials = {}
    j = 0

    for i in range(df_ev.shape[0]):
        trial_number = df_ev['TrialNumber'][i]

        trial_onset = df_ev['onset'][i]
        trial_offset = trial_onset + df_ev['duration'][i]

        trial_confs = []
        while j < len(vals_times) and vals_times[j] < trial_offset:
            if vals_times[j] > trial_onset:
                trial_confs.append(vals_conf[j])
            j += 1

        num_gaze = len(trial_confs)
        if num_gaze > 0:
            confRatio = np.sum(np.array(trial_confs) > conf_thresh)/num_gaze
            gazeconf_per_trials[trial_number] = (confRatio, num_gaze)
        else:
            gazeconf_per_trials[trial_number] = (np.nan, 0)

    df_ev[f'gaze_confidence_ratio_cThresh{conf_thresh}'] = df_ev.apply(lambda row: gazeconf_per_trials[row['TrialNumber']][0], axis=1)
    df_ev[f'gaze_count_cThresh{conf_thresh}'] = df_ev.apply(lambda row: gazeconf_per_trials[row['TrialNumber']][1], axis=1)

    return df_ev


def median_clean(frame_times, dist_x, dist_y):
    '''
    Within bins of 1/100 the number of frames,
    select only frames where distance between gaze and deepgaze falls within 0.6 stdev of the median
    These frames most likely reflect when deepgaze and gaze "look" at the same thing
    '''
    jump = int(len(dist_x)/100)
    idx = 0
    gap = 0.6 # interval of distances included around median, in stdev

    filtered_times = []
    filtered_distx = []
    filtered_disty = []

    current_medx = np.median(np.array(dist_x)[idx:idx+jump])
    current_medy = np.median(np.array(dist_y)[idx:idx+jump])
    stdevx = np.std(np.array(dist_x)[idx:idx+jump])
    stdevy = np.std(np.array(dist_y)[idx:idx+jump])

    for i in range(len(dist_x)):
        if i > (idx + jump):
            idx += 1
            current_medx = np.median(np.array(dist_x)[idx:idx+jump])
            current_medy = np.median(np.array(dist_y)[idx:idx+jump])
            stdevx = np.std(np.array(dist_x)[idx:idx+jump])
            stdevy = np.std(np.array(dist_y)[idx:idx+jump])

        if dist_x[i] < current_medx + (gap*stdevx):
            if dist_x[i] > current_medx - (gap*stdevx):
                if dist_y[i] < current_medy + (gap*stdevy):
                    if dist_y[i] > current_medy - (gap*stdevx):
                        filtered_times.append(frame_times[i])
                        filtered_distx.append(dist_x[i])
                        filtered_disty.append(dist_y[i])

    return filtered_times, filtered_distx, filtered_disty


def apply_poly(ref_times, distances, degree, all_times, anchors = [150, 150]):
    '''
    Fit polynomial to a distribution, then export points along that polynomial curve
    that correspond to specific gaze time stamps

    The very begining and end of distribution are excluded for stability
    e.g., participants sometimes look off screen at movie onset & offset,
    while deepgaze is biased to the center when shown a black screen.
    This really throws off polynomials
    '''
    if degree == 1:
        p1, p0 = np.polyfit(ref_times[anchors[0]:-anchors[1]], distances[anchors[0]:-anchors[1]], 1)
        p_of_all = p1*(all_times) + p0

    elif degree == 2:
        p2, p1, p0 = np.polyfit(ref_times[anchors[0]:-anchors[1]], distances[anchors[0]:-anchors[1]], 2)
        p_of_all = p2*(all_times**2) + p1*(all_times) + p0

    elif degree == 3:
        p3, p2, p1, p0 = np.polyfit(ref_times[anchors[0]:-anchors[1]], distances[anchors[0]:-anchors[1]], 3)
        p_of_all = p3*(all_times**3) + p2*(all_times**2) + p1*(all_times) + p0

    elif degree == 4:
        p4, p3, p2, p1, p0 = np.polyfit(ref_times[anchors[0]:-anchors[1]], distances[anchors[0]:-anchors[1]], 4)
        p_of_all = p4*(all_times**4) + p3*(all_times**3) + p2*(all_times**2) + p1*(all_times) + p0

    return p_of_all


def bidsify_EToutput(row, out_path, conf_thresh):
    '''
    Implement drift correction on gaze position and export pupil and gaze data in bids-compliant format
    '''
    task = row['task']

    log_path = row['log_path']

    [sub, ses, fnum, task_type, run_num, appendix] = os.path.basename(row['events_path']).split('_')
    print(sub, ses, fnum, task_type, run_num)
    if not os.path.exists(f'{out_path}/DC_gaze/{sub}_{ses}_{run_num}_{fnum}_{task_type}_DCplot.png'):
        #try:
        if True:
            onset_time = get_onset_time(log_path, row['run'], task)

            run_event = pd.read_csv(row['events_path'], sep = '\t', header=0)
            run_gaze = np.load(row['gaze_path'], allow_pickle=True)['gaze2d']

            if row['use_lowThresh']==1.0:
                gaze_threshold = conf_thresh
            else:
                gaze_threshold = 0.9
            reset_gaze_list, all_vals, clean_vals  = reset_gaze_time(run_gaze, onset_time, gaze_threshold)
            # normalized position (x and y), time (s) from onset and confidence for all gaze
            all_x, all_y, all_times, all_conf = all_vals
            all_times_arr = np.array(all_times)
            # distance from central fixation point for all gaze above confidence threshold
            clean_dist_x, clean_dist_y, clean_times, clean_conf = clean_vals
            # distance from central fixation for high-confidence gaze captured during periods of fixation (between trials)
            fix_dist_x, fix_dist_y, fix_times, fix_conf = get_fixation_gaze(run_event, clean_dist_x, clean_dist_y, clean_times, clean_conf)

            # median filter removes gaze too far off from median gaze position within sliding window, for cleaner curves (remove non-fixation points)
            mf_fix_times, mf_fix_dist_x, mf_fix_dist_y = median_clean(fix_times, fix_dist_x, fix_dist_y)

            if row['use_lowThresh'] == 1.0:
                deg_x, deg_y = 1, 1 # keep polynomial simpler to guard against outliers
            else:
                deg_x, deg_y = 4, 4
            anchors = [0, 50]
            # fit polynomial through distance between fixation and target
            # and use it apply correction to all gaze (no confidence threshold applied)
            p_of_all_x = apply_poly(mf_fix_times, mf_fix_dist_x, deg_x, all_times_arr, anchors=anchors)
            all_x_aligned = np.array(all_x) - (p_of_all_x)

            p_of_all_y = apply_poly(mf_fix_times, mf_fix_dist_y, deg_y, all_times_arr, anchors=anchors)
            all_y_aligned = np.array(all_y) - (p_of_all_y)

            # Export drift-corrected gaze, realigned timestamps, and all other metrics (pupils, etc) to bids-compliant .tsv file
            # guidelines: https://bids-specification--1128.org.readthedocs.build/en/1128/modality-specific-files/eye-tracking.html#sidecar-json-document-_eyetrackjson
            outpath_gaze = os.path.join(out_path, sub, ses)

            col_names = ['eye_timestamp',
                         'eye1_x_coordinate', 'eye1_y_coordinate',
                         'eye1_confidence',
                         'eye1_x_coordinate_driftCorr', 'eye1_y_coordinate_driftCorr',
                         'eye1_pupil_x_coordinate', 'eye1_pupil_y_coordinate',
                         'eye1_pupil_diameter',
                         'eye1_pupil_ellipse_axes',
                         'eye1_pupil_ellipse_angle',
                         'eye1_pupil_ellipse_center'
                         ]
            final_gaze_list = []
            #df_gaze = pd.DataFrame(columns=col_names)

            assert len(reset_gaze_list) == len(all_x_aligned)
            for i in range(len(reset_gaze_list)):
                gaze_pt = reset_gaze_list[i]
                assert gaze_pt['reset_time'] == all_times[i]

                gaze_pt_data = [
                                gaze_pt['reset_time'], # in s
                                #round(gaze_pt['reset_time']*1000, 0), # int, in ms
                                gaze_pt['norm_pos'][0], gaze_pt['norm_pos'][1],
                                gaze_pt['confidence'],
                                all_x_aligned[i], all_y_aligned[i],
                                gaze_pt['base_data']['norm_pos'][0], gaze_pt['base_data']['norm_pos'][1],
                                gaze_pt['base_data']['diameter'],
                                gaze_pt['base_data']['ellipse']['axes'],
                                gaze_pt['base_data']['ellipse']['angle'],
                                gaze_pt['base_data']['ellipse']['center'],
                ]

                final_gaze_list.append(gaze_pt_data)
                #df_gaze = pd.concat([df_gaze, pd.DataFrame(np.array(gaze_pt_data).reshape(1, -1), columns=df_gaze.columns)], ignore_index=True)

            df_gaze = pd.DataFrame(np.array(final_gaze_list, dtype=object), columns=col_names)
            gfile_path = f'{outpath_gaze}/{sub}_{ses}_{task_type}_{run_num}_conf{gaze_threshold}_eyetrack.tsv.gz'
            if os.path.exists(gfile_path):
                # just in case session redone... (one case in sub-03)
                gfile_path = f'{outpath_gaze}/{sub}_{ses}_{task_type}_{fnum}_{run_num}_conf{gaze_threshold}_eyetrack.tsv.gz'
            df_gaze.to_csv(gfile_path, sep='\t', header=True, index=False, compression='gzip')

            '''
            # .npz alternative to .tsv for now to test the code...
            # concat w pandas is VERY ineffective
            final_gaze_list = []
            assert len(reset_gaze_list) == len(all_x_aligned)
            for i in range(len(reset_gaze_list)):
                gaze_pt = reset_gaze_list[i]
                assert gaze_pt['reset_time'] == all_times[i]
                gaze_pt['norm_pos_driftCorr'] = (all_x_aligned[i], all_y_aligned[i])
                final_gaze_list.append(gaze_pt)

            gfile_path = f'{outpath_gaze}/{sub}_{ses}_{task_type}_{run_num}_conf{gaze_threshold}_eyetrack.npz'
            if os.path.exists(gfile_path):
                # just in case session redone... (one case in sub-03)
                gfile_path = f'{outpath_gaze}/{sub}_{ses}_{task_type}_{fnum}_{run_num}_conf{gaze_threshold}_eyetrack.npz'
            np.savez(gfile_path, gaze2d = final_gaze_list)
            '''

            # for each trial, capture all gaze and derive % of above-threshold gaze, add metric to events file and save
            run_event = assign_gazeConf2trial(run_event, all_times, all_conf, conf_thresh=0.9)
            run_event = assign_gazeConf2trial(run_event, all_times, all_conf, conf_thresh=0.75)
            outpath_events = f'{out_path}/Events_files'
            Path(outpath_events).mkdir(parents=True, exist_ok=True)
            run_event.to_csv(f'{outpath_events}/{sub}_{ses}_{fnum}_{task_type}_{run_num}_events.tsv', sep='\t', header=True, index=False)

            # export some additional plots to visulize the gaze drift correction and general QC.
            outpath_fig = os.path.join(out_path, 'DC_gaze')
            Path(outpath_fig).mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(5, 1, figsize=(7, 17.5))
            plot_labels = ['gaze_x', 'gaze_y', 'pupil_x', 'pupil_x']

            axes[0].scatter(all_times, all_x, color='xkcd:blue', alpha=all_conf)
            axes[0].scatter(all_times, all_x_aligned, color='xkcd:orange', alpha=all_conf)
            axes[0].set_ylim(-2, 2)
            axes[0].set_xlim(0, 350)
            axes[0].set_title(f'{sub} {task_type} {ses} {run_num} gaze_x')

            axes[1].scatter(all_times, all_y, color='xkcd:blue', alpha=all_conf)
            axes[1].scatter(all_times, all_y_aligned, color='xkcd:orange', alpha=all_conf)
            axes[1].set_ylim(-2, 2)
            axes[1].set_xlim(0, 350)
            axes[1].set_title(f'{sub} {task_type} {ses} {run_num} gaze_y')

            axes[2].scatter(fix_times, fix_dist_x, color='xkcd:orange', s=20, alpha=0.4)
            axes[2].scatter(mf_fix_times, mf_fix_dist_x, s=20, alpha=0.4)
            axes[2].plot(all_times_arr, p_of_all_x, color="xkcd:red", linewidth=2)
            axes[2].set_ylim(-2, 2)
            axes[2].set_xlim(0, 350)
            axes[2].set_title(f'{sub} {task_type} {ses} {run_num} fix_distance_x')

            axes[3].scatter(fix_times, fix_dist_y, color='xkcd:orange', s=20, alpha=0.4)
            axes[3].scatter(mf_fix_times, mf_fix_dist_y, s=20, alpha=0.4)
            axes[3].plot(all_times_arr, p_of_all_y, color="xkcd:red", linewidth=2)
            axes[3].set_ylim(-2, 2)
            axes[3].set_xlim(0, 350)
            axes[3].set_title(f'{sub} {task_type} {ses} {run_num} fix_distance_y')

            axes[4].scatter(run_event['onset'].to_numpy()+2.0, run_event[f'gaze_confidence_ratio_cThresh{gaze_threshold}'].to_numpy())
            axes[4].set_ylim(-0.1, 1.1)
            axes[4].set_xlim(0, 350)
            axes[4].set_title(f'{sub} {task_type} {ses} {run_num} ratio >{str(gaze_threshold)} confidence per trial')

            fig.savefig(f'{outpath_fig}/{sub}_{ses}_{run_num}_{fnum}_{task_type}_DCplot.png')
            plt.close()
        #except:
        #    print('could not process')


def main():
    # e.g., (elm): /unf/eyetracker/neuromod/triplets/sourcedata
    in_path = args.in_path
    out_path = args.out_path

    phase = args.phase

    if phase == 'A':
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
        Step 3: manual QCing... rate quality of each run, log in spreadsheet
        Compile clean list of runs to drift correct and bids-format
        Save as QCed_file_list.tsv in "out_path" directory
        Load to identify valid runs to be processed with steps 4 and 5
        '''

    elif phase == 'B':
        '''
        Step 4: apply drift correction to gaze based on known fixations

        Triplets & Word familiariry task details here:
        https://github.com/courtois-neuromod/task_stimuli/blob/4d1e66bdb66b722eb25a886a0008e2668054e470/src/tasks/language.py#L115

        TRIPLETS: Fixation in the center of the screen: (0,0)
        Task begins with fixation until the first trial's onset (~6s from task onset)
        For each trial, the 3 words (triplets) appear from the trial's onset until onset + duration (4s);
        Then, a central fixation point is shown from the time of [onset + duration] until [onset + duration + ISI (varied) - 0.1s]

        WORD FAMILIARITY: Fixation in the center of the screen: (0,0)
        Task begins with fixation until the first trial's onset (~9s from task onset)
        For each trial, a single word appears centrally from the trial's onset until onset + duration (0.5s);
        Then, a central fixation point is shown from the time of [onset + duration] until [onset + duration + ISI (varied) - 0.1s]

        Note from log: it seems that the task onset is well synched to the MRI starting (launch eyetracking and task)
        GO signal can be used as run's "time 0" for gaze & pupils

        Step 5: export to tsv.gz format following bids extension guidelines
        - pupil size and confidence
        - gaze position (before and after correction) and confidence;
        - Add gaze position in different metrics? in pixels (on screen), and then in pixels (stimulus image), and then normalized screen
        - set timestamp to 0 = task onset, export in ms (integer)
        '''
        # load list of valid files (those deserialized and exported as npz in step 2 that passed QC)
        outpath_report = os.path.join(out_path, 'QC_gaze')
        Path(outpath_report).mkdir(parents=True, exist_ok=True)

        file_list = pd.read_csv(f'{outpath_report}/QCed_file_list.tsv', sep='\t', header=0)
        clean_list = file_list[file_list['DO_NOT_USE']!=1.0]
        clean_list['gaze_path'] = clean_list.apply(lambda row: create_gaze_path(row, out_path), axis=1)
        clean_list['events_path'] = clean_list.apply(lambda row: create_event_path(row, in_path), axis=1)
        clean_list['log_path'] = clean_list.apply(lambda row: create_event_path(row, in_path, log=True), axis=1)

        # implements steps 4 and 5 on each run
        conf_thresh = args.cthresh
        clean_list.apply(lambda row: bidsify_EToutput(row, out_path, conf_thresh), axis=1)


if __name__ == '__main__':
    sys.exit(main())
