import os, glob, sys, json
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='label, QC and bids-formats eye tracking dataset')
    parser.add_argument('--in_path', type=str, required=True, help='absolute path to directory that contains all data (sourcedata)')
    parser.add_argument('--run_dir', default='', type=str, help='absolute path to main code directory')
    parser.add_argument('--out_path', type=str, default='./test.tsv', help='absolute path to output file')
    args = parser.parse_args()

    return args


# Assign run number to subtasks for retino and fLoc based on order in which they were administered
# TODO: coordinate w Basile to make sure the run numbers match those of the bold files
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


def create_gaze_path(row, file_path):
    '''
    for each run, create path to deserialized gaze file
    '''
    s = row['subject']
    ses = row['session']
    task = row["task"]

    task_root = file_path.split('/')[-1]
    if task_root == 'mario3':
        task = 'task-mario3'

    return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["run"]}_{row["file_number"]}_{task}_gaze2D.npz'


def create_event_path(row, file_path, log=False):
    '''
    for each run, create path to events.tsv file
    '''
    s = row['subject']
    ses = row['session']
    if log:
        return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["file_number"]}.log'
    else:
        if row['task'] in ['task-bar', 'task-rings', 'task-wedges', 'task-flocdef', 'task-flocalt']:
            return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["file_number"]}_{row["task"]}_events.tsv'
        else:
            return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["file_number"]}_{row["task"]}_{row["run"]}_events.tsv'


def create_ip_path(row, file_path):
    '''
    for each run, create path to info.player.json file
    '''
    s = row['subject']
    ses = row['session']
    r = row['run']
    t = row['task']
    fnum = row['file_number']

    if row['task'] in ['task-bar', 'task-rings', 'task-wedges', 'task-flocdef', 'task-flocalt']:
        return f'{file_path}/{s}/{ses}/{s}_{ses}_{fnum}.pupil/{t}/000/info.player.json'
    else:
        return f'{file_path}/{s}/{ses}/{s}_{ses}_{fnum}.pupil/{t}_{r}/000/info.player.json'


def get_onset_time(log_path, run_num, ip_path, gz_ts):
    onset_time_dict = {}
    TTL_0 = -1
    has_lines = True

    with open(log_path) as f:
        lines = f.readlines()
        if len(lines) == 0:
            has_lines = False
        for line in lines:
            if "fMRI TTL 0" in line:
                TTL_0 = line.split('\t')[0]
            elif "saved wide-format data to /scratch/neuromod/data" in line:
                rnum = line.split('\t')[-1].split('_')[-2]
                onset_time_dict[rnum] = float(TTL_0)
            elif "class 'src.tasks.videogame.VideoGameMultiLevel'" in line:
                rnum = line.split(': ')[-2].split('_')[-1]
                onset_time_dict[rnum] = float(TTL_0)
            elif "class 'src.tasks.localizers.FLoc'" in line:
                rnum = run2task_mapping['floc'][line.split(': ')[-2]]
                onset_time_dict[rnum] = float(TTL_0)
            elif "class 'src.tasks.retinotopy.Retinotopy'" in line:
                rnum = run2task_mapping['retino'][line.split(': ')[-2]]
                onset_time_dict[rnum] = float(TTL_0)

    if has_lines:
        o_time = onset_time_dict[run_num]

        with open(ip_path, 'r') as f:
            iplayer = json.load(f)
        sync_ts = iplayer['start_time_synced_s']
        syst_ts = iplayer['start_time_system_s']

        is_sync_gz = (gz_ts-sync_ts)**2 < (gz_ts-syst_ts)**2
        is_sync_ot = (o_time-sync_ts)**2 < (o_time-syst_ts)**2
        if is_sync_gz != is_sync_ot:
            if is_sync_ot:
                o_time += (syst_ts - sync_ts)
            else:
                o_time += (sync_ts - syst_ts)
    else:
        print('empty log file, onset time estimated from gaze timestamp')
        o_time = gz_ts

    return o_time


def reset_gaze_time(gaze, onset_time):
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

    return reset_gaze_list, (all_x, all_y, all_times, all_conf)


def get_fixation_gaze(df_ev, clean_dist_x, clean_dist_y, clean_times, task, med_fix=False, gap=0.6):
    '''
    Identify gaze that correspond to periods of fixation
    if med_fix, export median gaze position for each fixation

    Otherwise, export gaze that fall within +- 0.6 stdev of the median in x and y during a fixation
    '''
    fix_dist_x = []
    fix_dist_y = []
    fix_times = []

    j = 0

    for i in range(df_ev.shape[0]):
        has_fixation = True
        if 'mario' in task:
            if df_ev['trial_type'][i] != 'fixation_dot':
                has_fixation = False

        if has_fixation:
            if task == 'task-thingsmemory' or 'mario' in task:
                fixation_onset = df_ev['onset'][i]
                fixation_offset = fixation_onset + df_ev['duration'][i]
                #trial_offset = fixation_offset

            elif task == 'task-emotionvideos':
                fixation_onset = df_ev['onset_fixation_flip'][i]
                fixation_offset = df_ev['onset_video_flip'][i]
                #trial_offset = fixation_offset + df_ev['total_duration'][i]

            elif task in ['task-wordsfamiliarity', 'task-triplets']:
                fixation_onset = df_ev['onset'][i] - 3.0 if i == 0 else df_ev['onset'][i-1] + df_ev['duration'][i-1]
                fixation_offset = df_ev['onset'][i]
                #trial_offset = fixation_offset + df_ev['duration'][i]

            # add gaze from pre-trial fixation period
            trial_fd_x = []
            trial_fd_y = []
            trial_ftimes = []
            while j < len(clean_times) and clean_times[j] < fixation_offset:
                # + 0.8 = 800ms (0.8s) after trial offset to account for saccade
                if clean_times[j] > (fixation_onset + 0.8) and clean_times[j] < (fixation_offset - 0.1):
                    trial_fd_x.append(clean_dist_x[j])
                    trial_fd_y.append(clean_dist_y[j])
                    trial_ftimes.append(clean_times[j])
                j += 1

            if len(trial_fd_x) > 0:
                med_x = np.median(trial_fd_x)
                med_y = np.median(trial_fd_y)
                if med_fix:
                    fix_dist_x.append(med_x)
                    fix_dist_y.append(med_y)
                    fix_times.append(trial_ftimes[0])
                else:
                    stdevx= np.std(trial_fd_x)
                    stdevy= np.std(trial_fd_y)
                    trial_fd_x = np.array(trial_fd_x)
                    trial_fd_y = np.array(trial_fd_y)
                    trial_ftimes = np.array(trial_ftimes)
                    f1 = trial_fd_x < med_x + (gap*stdevx)
                    f2 = trial_fd_x > med_x - (gap*stdevx)
                    f3 = trial_fd_y < med_y + (gap*stdevy)
                    f4 = trial_fd_y > med_y - (gap*stdevy)
                    gaze_filter = (f1*f2*f3*f4).astype(bool)
                    if np.sum(gaze_filter) > 0:
                        fix_dist_x += trial_fd_x[gaze_filter].tolist()
                        fix_dist_y += trial_fd_y[gaze_filter].tolist()
                        fix_times += trial_ftimes[gaze_filter].tolist()

    return fix_dist_x, fix_dist_y, fix_times


def assign_gazeConf2trial(df_ev, vals_times, vals_conf, task, conf_thresh=0.9, add_count=True):

    gazeconf_per_trials = {}
    j = 0

    for i in range(df_ev.shape[0]):
        trial_number = df_ev['TrialNumber'][i]
        if task == 'task-emotionvideos':
            trial_onset = df_ev['onset_video_flip'][i]
            trial_offset = trial_onset + df_ev['total_duration'][i]

        elif task in ['task-thingsmemory', 'task-wordsfamiliarity', 'task-triplets']:
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
    if add_count:
        df_ev['gaze_count'] = df_ev.apply(lambda row: gazeconf_per_trials[row['TrialNumber']][1], axis=1)

    return df_ev


def assign_Compliance2trial(df_ev, vals_times, vals_x, vals_y, task, deg_va=1):

    fixCompliance_per_trials = {}
    j = 0

    for i in range(df_ev.shape[0]):
        trial_number = df_ev['TrialNumber'][i]

        if task == 'task-thingsmemory':
            fixation_onset = df_ev['onset'][i]
            fixation_offset = fixation_onset + df_ev['duration'][i]
            trial_offset = fixation_offset
            gaze_count = df_ev['gaze_count'][i]

        elif task == 'task-emotionvideos':
            fixation_onset = df_ev['onset_fixation_flip'][i]
            fixation_offset = df_ev['onset_video_flip'][i]
            trial_offset = fixation_offset + df_ev['total_duration'][i]

        elif task in ['task-wordsfamiliarity', 'task-triplets']:
            fixation_onset = 3.0 if i == 0 else df_ev['onset'][i-1] + df_ev['duration'][i-1]
            fixation_offset = df_ev['onset'][i]
            trial_offset = fixation_offset + df_ev['duration'][i]

        trial_comp = []
        while j < len(vals_times) and vals_times[j] < trial_offset:
            if vals_times[j] > fixation_onset and vals_times[j] < fixation_offset:
                # is gaze within 1 degree of visual angle of central fixation in x and y?
                x_comp = abs(vals_x[j] - 0.5) < (deg_va/17.5)
                y_comp = abs(vals_y[j] - 0.5) < (deg_va/14.0)
                trial_comp.append(x_comp and y_comp)
            j += 1

        num_gaze = len(trial_comp)
        if task == 'task-thingsmemory':
            assert gaze_count == num_gaze
        fixCompliance_per_trials[trial_number] = np.sum(trial_comp)/num_gaze if num_gaze > 0 else np.nan

    df_ev[f'fixation_compliance_ratio_deg{deg_va}'] = df_ev.apply(lambda row: fixCompliance_per_trials[row['TrialNumber']], axis=1)

    return df_ev


def assign_gzMetrics2trial_mario(df_ev, vals_times, vals_conf, vals_x, vals_y, conf_thresh=0.9, add_count=True):

    bk2_times = {}
    bk2_name = None
    bk2_onset, bk2_offset = -1, -1

    for i in range(df_ev.shape[0]):
        if df_ev['trial_type'][i] == 'fixation_dot' and bk2_onset > 0:
            bk2_offset = df_ev['onset'][i]
            bk2_times[bk2_name] = [bk2_onset, bk2_offset]
        elif df_ev['trial_type'][i] == 'gym-retro_game':
            bk2_onset = df_ev['onset'][i]
            bk2_name = df_ev['stim_file'][i]

    gaze_confidence = []
    fix_compliance = [[], [], [], []]
    #fix_compliance_2 = []
    #fix_compliance_3 = []
    gaze_per_trial = []

    j = 0
    for i in range(df_ev.shape[0]):
        trial_onset = df_ev['onset'][i]
        trial_type = df_ev['trial_type'][i]

        if trial_type == 'fixation_dot':
            trial_comp = [[], [], [], []]
            #trial_comp_2 = []
            #trial_comp_3 = []
            trial_conf = []
            trial_offset = trial_onset + df_ev['duration'][i]

            while j < len(vals_times) and vals_times[j] < trial_offset:
                if vals_times[j] > trial_onset:
                    # is gaze within 1, 2, 3 degrees of visual angle of central fixation in x and y?
                    for k, deg_val in enumerate([0.5, 1, 2, 3]):
                        x_comp = abs(vals_x[j] - 0.5) < (deg_val/17.5)
                        y_comp = abs(vals_y[j] - 0.5) < (deg_val/14.0)
                        trial_comp[k].append(x_comp and y_comp)
                    trial_conf.append(vals_conf[j] > conf_thresh)
                j += 1
            num_gaze = len(trial_comp_1)
            if num_gaze > 0:
                for k, deg_val in enumerate([0.5, 1, 2, 3]):
                    fix_compliance[k].append(np.sum(trial_comp[k])/num_gaze)
                gaze_per_trial.append(num_gaze)
                gaze_confidence.append(np.sum(trial_conf)/num_gaze)
            else:
                for k, deg_val in enumerate([0.5, 1, 2, 3]):
                    fix_compliance[k].append(np.nan)
                gaze_per_trial.append(np.nan)
                gaze_confidence.append(np.nan)


        elif trial_type == 'gym-retro_game':
            trial_conf = []
            trial_offset = bk2_times[df_ev['stim_file'][i]][1]
            assert trial_onset == bk2_times[df_ev['stim_file'][i]][0]

            while j < len(vals_times) and vals_times[j] < trial_offset:
                if vals_times[j] > trial_onset:
                    trial_conf.append(vals_conf[j] > conf_thresh)
                j += 1

            num_gaze = len(trial_conf)
            if num_gaze > 0:
                gaze_confidence.append(np.sum(trial_conf)/num_gaze)
                gaze_per_trial.append(num_gaze)
            else:
                gaze_confidence.append(np.nan)
                gaze_per_trial.append(np.nan)
            for k, deg_val in enumerate([0.5, 1, 2, 3]):
                fix_compliance[k].append(np.nan)

        else:
            gaze_confidence.append(np.nan)
            gaze_per_trial.append(np.nan)
            for k, deg_val in enumerate([0.5, 1, 2, 3]):
                fix_compliance[k].append(np.nan)

    # Insert 3 new columns in df_ev
    df_ev.insert(loc=df_ev.shape[1]-2, column=f'gaze_confidence_ratio_cThresh{conf_thresh}',
                 value=gaze_confidence, allow_duplicates=True)
    if add_count:
        df_ev.insert(loc=df_ev.shape[1]-2, column='gaze_count',
                     value=gaze_per_trial, allow_duplicates=True)
        for k, deg_val in enumerate([0.5, 1, 2, 3]):
            df_ev.insert(loc=df_ev.shape[1]-2, column=f'fixation_compliance_ratio_deg{deg_val}',
                         value=fix_compliance[k], allow_duplicates=True)

    return df_ev


def driftcorr_fromlast(fd_x, fd_y, f_times, all_x, all_y, all_times):
    i = 0
    j = 0
    all_x_aligned = []
    all_y_aligned = []

    for i in range(len(all_times)):
        if j < len(f_times)-1 and all_times[i] > f_times[j+1]:
            j += 1
        all_x_aligned.append(all_x[i] - fd_x[j])
        all_y_aligned.append(all_y[i] - fd_y[j])

    return all_x_aligned, all_y_aligned


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


def bidsify_EToutput(row, out_path):

    task_root = out_path.split('/')[-1]

    if task_root in ['retino', 'floc']:
        skip_run_num = True
        [sub, ses, fnum, task_type, appendix] = os.path.basename(row['events_path']).split('_')
        run_num = run2task_mapping[task_root][task_type]
    else:
        skip_run_num = False
        [sub, ses, fnum, task_type, run_num, appendix] = os.path.basename(row['events_path']).split('_')

    pseudo_task = 'task-mario3' if task_root == 'mario3' else task_type
    print(sub, ses, fnum, pseudo_task, run_num)

    bids_out_path = f'{out_path}/final_bids/{sub}/{ses}'
    Path(bids_out_path).mkdir(parents=True, exist_ok=True)
    gfile_path = f'{bids_out_path}/{sub}_{ses}_{pseudo_task}_{run_num}_eyetrack.tsv.gz'
    if os.path.exists(gfile_path):
        # just in case session's run is done twice... note: not bids...
        gfile_path = f'{bids_out_path}/{sub}_{ses}_{pseudo_task}_{fnum}_{run_num}_eyetrack.tsv.gz'

    #if True:
    try:
        run_event = pd.read_csv(row['events_path'], sep = '\t', header=0)
        run_gaze = np.load(row['gaze_path'], allow_pickle=True)['gaze2d']

        # identifies logged run start time (mri TTL 0) on clock that matches the gaze using info.player.json
        onset_time = get_onset_time(row['log_path'], row['run'], row['infoplayer_path'], run_gaze[10]['timestamp'])

        reset_gaze_list, all_vals  = reset_gaze_time(run_gaze, onset_time)
        # normalized position (x and y), time (s) from onset and confidence for all gaze
        all_x, all_y, all_times, all_conf = all_vals
        all_times_arr = np.array(all_times)

        # Export uncorrected gaze, realigned timestamps, and all other metrics (pupils, etc) to bids-compliant .tsv file
        # guidelines: https://bids-specification--1128.org.readthedocs.build/en/1128/modality-specific-files/eye-tracking.html#sidecar-json-document-_eyetrackjson
        col_names = ['eye_timestamp',
                     'eye1_x_coordinate', 'eye1_y_coordinate',
                     'eye1_confidence',
                     'eye1_pupil_x_coordinate', 'eye1_pupil_y_coordinate',
                     'eye1_pupil_diameter',
                     'eye1_pupil_ellipse_axes',
                     'eye1_pupil_ellipse_angle',
                     'eye1_pupil_ellipse_center'
                     ]

        final_gaze_list = []

        assert len(reset_gaze_list) == len(all_x)
        for i in range(len(reset_gaze_list)):
            gaze_pt = reset_gaze_list[i]
            assert gaze_pt['reset_time'] == all_times[i]

            gaze_pt_data = [
                            gaze_pt['reset_time'], # in s
                            #round(gaze_pt['reset_time']*1000, 0), # int, in ms
                            gaze_pt['norm_pos'][0], gaze_pt['norm_pos'][1],
                            gaze_pt['confidence'],
                            gaze_pt['base_data']['norm_pos'][0], gaze_pt['base_data']['norm_pos'][1],
                            gaze_pt['base_data']['diameter'],
                            gaze_pt['base_data']['ellipse']['axes'],
                            gaze_pt['base_data']['ellipse']['angle'],
                            gaze_pt['base_data']['ellipse']['center'],
            ]

            final_gaze_list.append(gaze_pt_data)

        df_gaze = pd.DataFrame(np.array(final_gaze_list, dtype=object), columns=col_names)
        df_gaze.to_csv(gfile_path, sep='\t', header=True, index=False, compression='gzip')

    except:
        print('could not process')


def main():
    '''
    This script exports raw bids-compliant gaze and pupil metrics (no drift correction in .tsv.gz format,
    according to the following proposed bids extension guidelines:
    https://bids-specification--1128.org.readthedocs.build/en/1128/modality-specific-files/eye-tracking.html#sidecar-json-document-_eyetrackjson
    '''

    args = get_arguments()
    # e.g., (elm): /unf/eyetracker/neuromod/triplets/sourcedata
    in_path = args.in_path
    out_path = args.out_path

    # load list of valid files (those deserialized and exported as npz in step 2 that passed QC)
    outpath_report = os.path.join(out_path, 'QC_gaze')
    Path(outpath_report).mkdir(parents=True, exist_ok=True)

    file_list = pd.read_csv(f'{outpath_report}/QCed_file_list.tsv', sep='\t', header=0)

    clean_list = file_list[file_list['DO_NOT_USE']!=1.0]

    clean_list['gaze_path'] = clean_list.apply(lambda row: create_gaze_path(row, out_path), axis=1)
    clean_list['events_path'] = clean_list.apply(lambda row: create_event_path(row, in_path), axis=1)
    clean_list['log_path'] = clean_list.apply(lambda row: create_event_path(row, in_path, log=True), axis=1)
    clean_list['infoplayer_path'] = clean_list.apply(lambda row: create_ip_path(row, in_path), axis=1)

    # implement drift correction on each run
    clean_list.apply(lambda row: bidsify_EToutput(row, out_path), axis=1)


if __name__ == '__main__':
    sys.exit(main())
