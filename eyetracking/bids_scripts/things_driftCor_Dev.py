import os, glob, sys, json
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='clean up, label, QC and bids-formats the triplets eye tracking dataset')
    parser.add_argument('--in_path', type=str, required=True, help='absolute path to directory that contains all data (sourcedata)')
    parser.add_argument('--phase_num', default=1, type=int, help='phase number')
    parser.add_argument('--out_path', type=str, required=True, help='absolute path to output directory')
    args = parser.parse_args()

    return args


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


def get_fixation_gaze_things(df_ev, clean_dist_x, clean_dist_y, clean_times, fix_period="image", med_fix=False, gap=0.6):
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

        if fix_period == 'image':
            fixation_onset = df_ev['onset'][i]
            fixation_offset = fixation_onset + df_ev['duration'][i]
            onset_buffer = 0.0

        elif fix_period == 'isi':
            fixation_onset = df_ev['onset'][i] + df_ev['duration'][i]
            fixation_offset = fixation_onset + 1.49
            onset_buffer = 0.6

        elif fix_period == 'image+isi':
            fixation_onset = df_ev['onset'][i]
            fixation_offset = fixation_onset + df_ev['duration'][i] + 1.49
            onset_buffer = 0.0


        # add gaze from trial fixation period
        trial_fd_x = []
        trial_fd_y = []
        trial_ftimes = []
        while j < len(clean_times) and clean_times[j] < fixation_offset:
            # + 0.8 = 800ms (0.8s) after trial offset to account for saccade
            # if clean_times[j] > (fixation_onset + 0.8) and clean_times[j] < (fixation_offset - 0.1):
            if clean_times[j] > (fixation_onset + onset_buffer) and clean_times[j] < (fixation_offset - 0.1):
                trial_fd_x.append(clean_dist_x[j])
                trial_fd_y.append(clean_dist_y[j])
                trial_ftimes.append(clean_times[j])
            j += 1

        #if len(trial_fd_x) > 0:
        if len(trial_fd_x) > 20:
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


def add_metrics_2events(df_ev,
                        all_times,
                        all_conf,
                        all_x,
                        all_y,
                        all_x_aligned,
                        all_y_aligned,
                        conf_thresh=0.9,
                        ):

    all_distances = get_distances(all_x_aligned, all_y_aligned)

    metrics_per_trials = {}
    all_idx = 0

    for i in range(df_ev.shape[0]):
        trial_number = df_ev['TrialNumber'][i]

        trial_onset = df_ev['onset'][i]
        trial_offset = trial_onset + df_ev['duration'][i]

        isi_onset = trial_offset + 0.6
        isi_offset = trial_offset + 1.49

        if i == 0:
            isi_confs = []
            isi_x = []
            isi_y = []
            isi_distances = []
            while all_idx < len(all_times) and all_times[all_idx] < trial_onset:
                if all_times[all_idx] > (trial_onset - 0.89):
                    isi_confs.append(all_conf[all_idx])
                    isi_x.append(all_x[all_idx])
                    isi_y.append(all_y[all_idx])
                    isi_distances.append(all_distances[all_idx])

                all_idx += 1

            conf_filter = np.array(isi_confs) > conf_thresh
            f_sum = np.sum(conf_filter)
            if f_sum:
                isi_x_arr = np.array(isi_x)[conf_filter]
                isi_y_arr = np.array(isi_y)[conf_filter]
                isi_dist_arr = np.array(isi_distances)[conf_filter]
            metrics_per_trials[trial_number-1] = {
                'isi_gaze_count': len(isi_confs),
                'isi_gaze_conf_90': np.sum(np.array(isi_confs) > 0.9)/len(isi_confs) if len(isi_confs) > 0 else np.nan,
                'isi_gaze_conf_75': np.sum(np.array(isi_confs) > 0.75)/len(isi_confs) if len(isi_confs) > 0 else np.nan,

                f'isi_stdev_x_{conf_thresh}': np.std(isi_x_arr) if f_sum else np.nan,
                f'isi_stdev_y_{conf_thresh}': np.std(isi_y_arr) if f_sum else np.nan,
                f'isi_median_x_{conf_thresh}': np.median(isi_x_arr) if f_sum else np.nan,
                f'isi_median_y_{conf_thresh}': np.median(isi_y_arr) if f_sum else np.nan,
                'isi_fix_compliance_ratio_deg0.5': np.sum(isi_dist_arr < 0.5)/f_sum if f_sum else np.nan,
                'isi_fix_compliance_ratio_deg1': np.sum(isi_dist_arr < 1.0)/f_sum if f_sum else np.nan,
                'isi_fix_compliance_ratio_deg2': np.sum(isi_dist_arr < 2.0)/f_sum if f_sum else np.nan,
                'isi_fix_compliance_ratio_deg3': np.sum(isi_dist_arr < 3.0)/f_sum if f_sum else np.nan,
                }

        trial_confs = []
        trial_x = []
        trial_y = []
        trial_distance = []
        isi_confs = []
        isi_x = []
        isi_y = []
        isi_distances = []

        while all_idx < len(all_times) and all_times[all_idx] < isi_offset:
            if all_times[all_idx] > trial_onset:
                if all_times[all_idx] < trial_offset:
                    trial_confs.append(all_conf[all_idx])
                    trial_x.append(all_x[all_idx])
                    trial_y.append(all_y[all_idx])
                    trial_distance.append(all_distances[all_idx])

                elif all_times[all_idx] > isi_onset:
                    isi_confs.append(all_conf[all_idx])
                    isi_x.append(all_x[all_idx])
                    isi_y.append(all_y[all_idx])
                    isi_distances.append(all_distances[all_idx])

            all_idx += 1

        t_conf_filter = np.array(trial_confs) > conf_thresh
        t_f_sum = np.sum(t_conf_filter)
        if t_f_sum:
            trial_x_arr = np.array(trial_x)[t_conf_filter]
            trial_y_arr = np.array(trial_y)[t_conf_filter]
            trial_dist_arr = np.array(trial_distance)[t_conf_filter]

        i_conf_filter = np.array(isi_confs) > conf_thresh
        i_f_sum = np.sum(i_conf_filter)
        if i_f_sum:
            isi_x_arr = np.array(isi_x)[i_conf_filter]
            isi_y_arr = np.array(isi_y)[i_conf_filter]
            isi_dist_arr = np.array(isi_distances)[i_conf_filter]
        metrics_per_trials[trial_number] = {
            'trial_gaze_count': len(trial_confs),
            'trial_gaze_conf_90': np.sum(np.array(trial_confs) > 0.9)/len(trial_confs) if len(trial_confs) > 0 else np.nan,
            'trial_gaze_conf_75': np.sum(np.array(trial_confs) > 0.75)/len(trial_confs) if len(trial_confs) > 0 else np.nan,
            'isi_gaze_count': len(isi_confs),
            'isi_gaze_conf_90': np.sum(np.array(isi_confs) > 0.9)/len(isi_confs) if len(isi_confs) > 0 else np.nan,
            'isi_gaze_conf_75': np.sum(np.array(isi_confs) > 0.75)/len(isi_confs) if len(isi_confs) > 0 else np.nan,

            f'trial_stdev_x_{conf_thresh}': np.std(trial_x_arr) if t_f_sum else np.nan,
            f'trial_stdev_y_{conf_thresh}': np.std(trial_y_arr) if t_f_sum else np.nan,
            'trial_fix_compliance_ratio_deg0.5': np.sum(trial_dist_arr < 0.5)/t_f_sum if t_f_sum else np.nan,
            'trial_fix_compliance_ratio_deg1': np.sum(trial_dist_arr < 1.0)/t_f_sum if t_f_sum else np.nan,
            'trial_fix_compliance_ratio_deg2': np.sum(trial_dist_arr < 2.0)/t_f_sum if t_f_sum else np.nan,
            'trial_fix_compliance_ratio_deg3': np.sum(trial_dist_arr < 3.0)/t_f_sum if t_f_sum else np.nan,

            f'isi_stdev_x_{conf_thresh}': np.std(isi_x_arr) if i_f_sum else np.nan,
            f'isi_stdev_y_{conf_thresh}': np.std(isi_y_arr) if i_f_sum else np.nan,
            f'isi_median_x_{conf_thresh}': np.median(isi_x_arr) if i_f_sum else np.nan,
            f'isi_median_y_{conf_thresh}': np.median(isi_y_arr) if i_f_sum else np.nan,
            'isi_fix_compliance_ratio_deg0.5': np.sum(isi_dist_arr < 0.5)/i_f_sum if i_f_sum else np.nan,
            'isi_fix_compliance_ratio_deg1': np.sum(isi_dist_arr < 1.0)/i_f_sum if i_f_sum else np.nan,
            'isi_fix_compliance_ratio_deg2': np.sum(isi_dist_arr < 2.0)/i_f_sum if i_f_sum else np.nan,
            'isi_fix_compliance_ratio_deg3': np.sum(isi_dist_arr < 3.0)/i_f_sum if i_f_sum else np.nan,
            }

    '''
    Potentially: correction will be anchored on fixation during preceeding ISI (ISI_0)
        TODO: insert trialwise QC metrics in events file to evaluate quality of fixation
        - gaze confidence ratio (of 0.9, 0.75) during image (signal quality)
        - gaze confidence ratio during preceding ISI (ISI_0)
        - gaze confidence ratio during following ISI (ISI_1)
        - gaze count during image, ISI_0 and ISI_1 (camera freeze?)
        - stdev in x and y during image : good fixation?
        - stdev in x and y during ISI_0, ISI_0 : reliable fixations?
        - distance in deg of visual angle between medians in ISI_0 and ISI_1: moved?
        - drift correction confidence index:
            IF high confidence and gaze count at ISI_0, ISI_1 and image
            IF low stdev in x and y at ISI_0 and ISI_1
            IF low distance (in deg v angle) between medians between ISI_0 and ISI_1 (no head mvt)
            then high confidence in precision of drift correction for that trial
        - fixation compliance ratio:
            assuming high drift corr conf index, compute ratio of
            gaze positioned within 1, 2 or 3 deg of vis angle from central fixation during image presentation
    '''
    #print(metrics_per_trials.keys(), metrics_per_trials[0].keys(), metrics_per_trials[1].keys(), metrics_per_trials)
    '''
    Insert gaze count: pre-isi, image presentation and post-isi
    '''
    df_ev['pre-isi_gaze_count_ratio'] = df_ev.apply(lambda row: (metrics_per_trials[row['TrialNumber']-1]['isi_gaze_count'])/(250*0.89), axis=1)
    df_ev['trial_gaze_count_ratio'] = df_ev.apply(lambda row: (metrics_per_trials[row['TrialNumber']]['trial_gaze_count'])/(250*2.98), axis=1)
    #df_ev[f'post-isi_gaze_count_ratio'] = df_ev.apply(lambda row: (metrics_per_trials[row['TrialNumber']]['isi_gaze_count'])/(250*0.74), axis=1)

    '''
    Insert gaze confidence ratio, out of all collected gaze (0.9 and 0.75 thresholds): pre-isi, image presentation and post-isi
    '''
    df_ev['pre-isi_gaze_confidence_ratio_0.9'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']-1]['isi_gaze_conf_90'], axis=1)
    df_ev['pre-isi_gaze_confidence_ratio_0.75'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']-1]['isi_gaze_conf_75'], axis=1)
    df_ev['trial_gaze_confidence_ratio_0.9'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['trial_gaze_conf_90'], axis=1)
    df_ev['trial_gaze_confidence_ratio_0.75'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['trial_gaze_conf_75'], axis=1)
    #df_ev[f'post-isi_gaze_confidence_ratio_0.9'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['isi_gaze_conf_90'], axis=1)
    #df_ev[f'post-isi_gaze_confidence_ratio_0.75'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['isi_gaze_conf_75'], axis=1)

    '''
    Insert stdev (in x and y, in normalized position): pre-isi, image presentation and post-isi
    '''
    df_ev[f'pre-isi_stdev_x_{conf_thresh}'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']-1][f'isi_stdev_x_{conf_thresh}'], axis=1)
    df_ev[f'pre-isi_stdev_y_{conf_thresh}'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']-1][f'isi_stdev_y_{conf_thresh}'], axis=1)
    df_ev[f'trial_stdev_x_{conf_thresh}'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']][f'trial_stdev_x_{conf_thresh}'], axis=1)
    df_ev[f'trial_stdev_y_{conf_thresh}'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']][f'trial_stdev_y_{conf_thresh}'], axis=1)
    #df_ev[f'post-isi_stdev_x_{conf_thresh}'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']][f'isi_stdev_x_{conf_thresh}'], axis=1)
    #df_ev[f'post-isi_stdev_y_{conf_thresh}'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']][f'isi_stdev_y_{conf_thresh}'], axis=1)

    '''
    Insert distance between median positions, in deg of visual angle, between pre- and pos-isi (excessive head motion)
    '''
    df_ev['pre-post-isi_distance_in_deg'] = df_ev.apply(lambda row: get_isi_distance(metrics_per_trials, row['TrialNumber'], conf_thresh), axis=1)

    '''
    Insert fixation compliance ratios
    '''
    df_ev['pre-isi_fixation_compliance_ratio_0.5'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']-1]['isi_fix_compliance_ratio_deg0.5'], axis=1)
    df_ev['pre-isi_fixation_compliance_ratio_1.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']-1]['isi_fix_compliance_ratio_deg1'], axis=1)
    df_ev['pre-isi_fixation_compliance_ratio_2.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']-1]['isi_fix_compliance_ratio_deg2'], axis=1)
    df_ev['pre-isi_fixation_compliance_ratio_3.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']-1]['isi_fix_compliance_ratio_deg3'], axis=1)

    df_ev['trial_fixation_compliance_ratio_0.5'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['trial_fix_compliance_ratio_deg0.5'], axis=1)
    df_ev['trial_fixation_compliance_ratio_1.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['trial_fix_compliance_ratio_deg1'], axis=1)
    df_ev['trial_fixation_compliance_ratio_2.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['trial_fix_compliance_ratio_deg2'], axis=1)
    df_ev['trial_fixation_compliance_ratio_3.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['trial_fix_compliance_ratio_deg3'], axis=1)

    #df_ev[f'post-isi_fixation_compliance_ratio_0.5'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['isi_fix_compliance_ratio_deg0.5'], axis=1)
    #df_ev[f'post-isi_fixation_compliance_ratio_1.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['isi_fix_compliance_ratio_deg1'], axis=1)
    #df_ev[f'post-isi_fixation_compliance_ratio_2.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['isi_fix_compliance_ratio_deg2'], axis=1)
    #df_ev[f'post-isi_fixation_compliance_ratio_3.0'] = df_ev.apply(lambda row: metrics_per_trials[row['TrialNumber']]['isi_fix_compliance_ratio_deg3'], axis=1)


    return df_ev


def driftcorr_fromlast(fd_x, fd_y, f_times, all_x, all_y, all_times):
    i = 0
    j = 0
    all_x_aligned = []
    all_y_aligned = []

    for i in range(len(all_times)):
        while j < len(f_times)-1 and all_times[i] > f_times[j+1]:
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


def get_distances(x, y, is_distance=False):
    '''
    if is_distance == True:
        x and y are relative distances from center, else they are normalized coordinates
    '''
    assert len(x) == len(y)

    dist_in_pix = 4164 # in pixels
    m_vecpos = np.array([0., 0., dist_in_pix])

    all_pos = np.stack((x, y), axis=1)
    if is_distance:
        gaze = (all_pos - 0.0)*(1280, 1024)
    else:
        gaze = (all_pos - 0.5)*(1280, 1024)
    gaze_vecpos = np.concatenate((gaze, np.repeat(dist_in_pix, len(gaze)).reshape((-1, 1))), axis=1)

    all_distances = []
    for gz_vec in gaze_vecpos:
        vectors = np.stack((m_vecpos, gz_vec), axis=0)
        distance = np.rad2deg(np.arccos(1.0 - pdist(vectors, metric='cosine')))[0]
        all_distances.append(distance)

    return all_distances


def get_isi_distance(metrics_dict, trial_num, conf_thresh):

    pre_x = (metrics_dict[trial_num-1][f'isi_median_x_{conf_thresh}'] - 0.5)*1280
    pre_y = (metrics_dict[trial_num-1][f'isi_median_y_{conf_thresh}'] - 0.5)*1024
    post_x = (metrics_dict[trial_num][f'isi_median_x_{conf_thresh}'] - 0.5)*1280
    post_y = (metrics_dict[trial_num][f'isi_median_y_{conf_thresh}'] - 0.5)*1024

    dist_in_pix = 4164 # in pixels

    vectors = np.array([[pre_x, pre_y, dist_in_pix], [post_x, post_y, dist_in_pix]])
    distance = np.rad2deg(np.arccos(1.0 - pdist(vectors, metric='cosine')))[0]

    return distance


def get_trial_distances(df_ev, x, y, times, confs):
    '''
    Reset gaze time in relation to trial onset
    Calculate distance from center (0.5, 0.5) for each trial gaze

    Export within-trial position (x and y), distance to center, relative time stamp and confidence
    '''

    assert len(x) == len(y)
    assert len(x) == len(times)
    assert len(x) == len(confs)

    dist_in_pix = 4164 # in pixels
    m_vecpos = np.array([0., 0., dist_in_pix])

    all_dist = []
    all_x = []
    all_y = []
    all_times = []
    all_confs = []

    j = 0

    for i in range(df_ev.shape[0]):
        trial_onset = df_ev['onset'][i]
        trial_offset = trial_onset + df_ev['duration'][i] + 1.49

        # add gaze from trial period
        trial_pos = []
        trial_times = []
        trial_confs = []
        while j < len(times) and times[j] < trial_offset:
            if times[j] > trial_onset and times[j] < trial_offset:
                trial_pos.append((x[j], y[j]))
                trial_times.append(times[j] - trial_onset)
                trial_confs.append(confs[j])
            j += 1

        if len(trial_pos) > 0:
            t_pos = np.array(trial_pos)
            gaze = (t_pos - 0.5)*(1280, 1024)
            gaze_vecpos = np.concatenate((gaze, np.repeat(dist_in_pix, len(gaze)).reshape((-1, 1))), axis=1)

            trial_distances = []
            for gz_vec in gaze_vecpos:
                vectors = np.stack((m_vecpos, gz_vec), axis=0)
                distance = np.rad2deg(np.arccos(1.0 - pdist(vectors, metric='cosine')))[0]
                trial_distances.append(distance)

            all_dist += trial_distances
            all_x += t_pos[:, 0].tolist()
            all_y += t_pos[:, 1].tolist()
            all_times += trial_times
            all_confs += trial_confs

    return all_dist, all_x, all_y, all_times, all_confs


def get_trial_distances_plusMetrics(df_ev, strategy, x, y, times, confs, conf_thresh, filter=False):
    '''
    Reset gaze time in relation to trial onset
    Calculate distance from center (0.5, 0.5) for each trial gaze

    Export within-trial position (x and y), distance to center, relative time stamp and confidence
    '''

    assert len(x) == len(y)
    assert len(x) == len(times)
    assert len(x) == len(confs)

    dist_in_pix = 4164 # in pixels
    m_vecpos = np.array([0., 0., dist_in_pix])

    if filter:
        gz_filter = np.array(confs) > conf_thresh
        x = np.array(x)[gz_filter].tolist()
        y = np.array(y)[gz_filter].tolist()
        times = np.array(times)[gz_filter].tolist()
        confs = np.array(confs)[gz_filter].tolist()

    all_dist = []
    all_x = []
    all_y = []
    all_times = []
    all_confs = []

    fix_gz_count = []
    fix_conf_ratio = []
    fix_stdev_x = []
    fix_stdev_y = []
    pre_post_isi_dist = []
    trial_fixCom = []

    j = 0

    for i in range(df_ev.shape[0]):
        trial_onset = df_ev['onset'][i]
        trial_offset = trial_onset + df_ev['duration'][i] + 1.49

        # add gaze from trial period
        trial_pos = []
        trial_times = []
        trial_confs = []
        while j < len(times) and times[j] < trial_offset:
            if times[j] > trial_onset and times[j] < trial_offset:
                trial_pos.append((x[j], y[j]))
                trial_times.append(times[j] - trial_onset)
                trial_confs.append(confs[j])
            j += 1

        if len(trial_pos) > 0:
            t_pos = np.array(trial_pos)
            gaze = (t_pos - 0.5)*(1280, 1024)
            gaze_vecpos = np.concatenate((gaze, np.repeat(dist_in_pix, len(gaze)).reshape((-1, 1))), axis=1)

            trial_distances = []
            for gz_vec in gaze_vecpos:
                vectors = np.stack((m_vecpos, gz_vec), axis=0)
                distance = np.rad2deg(np.arccos(1.0 - pdist(vectors, metric='cosine')))[0]
                trial_distances.append(distance)

            all_dist += trial_distances
            all_x += t_pos[:, 0].tolist()
            all_y += t_pos[:, 1].tolist()
            all_times += trial_times
            all_confs += trial_confs

            if strategy == 'isi':
                fix_gz_count += np.repeat(df_ev['pre-isi_gaze_count_ratio'][i], len(trial_pos)).tolist()
                fix_conf_ratio += np.repeat(df_ev['pre-isi_gaze_confidence_ratio_0.9'][i], len(trial_pos)).tolist()
                fix_stdev_x += np.repeat(df_ev[f'pre-isi_stdev_x_{conf_thresh}'][i], len(trial_pos)).tolist()
                fix_stdev_y += np.repeat(df_ev[f'pre-isi_stdev_y_{conf_thresh}'][i], len(trial_pos)).tolist()
                pre_post_isi_dist += np.repeat(df_ev['pre-post-isi_distance_in_deg'][i], len(trial_pos)).tolist()
                trial_fixCom += np.repeat(df_ev['trial_fixation_compliance_ratio_1.0'][i], len(trial_pos)).tolist()
            else:
                fix_gz_count += np.repeat(df_ev['trial_gaze_count_ratio'][i], len(trial_pos)).tolist()
                fix_conf_ratio += np.repeat(df_ev['trial_gaze_confidence_ratio_0.9'][i], len(trial_pos)).tolist()
                fix_stdev_x += np.repeat(df_ev[f'trial_stdev_x_{conf_thresh}'][i], len(trial_pos)).tolist()
                fix_stdev_y += np.repeat(df_ev[f'trial_stdev_y_{conf_thresh}'][i], len(trial_pos)).tolist()
                pre_post_isi_dist += np.repeat(df_ev['pre-post-isi_distance_in_deg'][i], len(trial_pos)).tolist()
                trial_fixCom += np.repeat(df_ev['trial_fixation_compliance_ratio_1.0'][i], len(trial_pos)).tolist()

    return ((all_dist, all_x, all_y, all_times, all_confs), (fix_gz_count, fix_conf_ratio, fix_stdev_x, fix_stdev_y, pre_post_isi_dist, trial_fixCom))


def driftCorr_ETtests(row, out_path, phase_num=1):

    task_root = out_path.split('/')[-1]

    [sub, ses, fnum, task_type, run_num, appendix] = os.path.basename(row['events_path']).split('_')
    print(sub, ses, fnum, task_type, run_num)

    if phase_num in [1, 2, 3, 4]:
        outpath_fig = os.path.join(out_path, 'TEST_gaze')
        Path(outpath_fig).mkdir(parents=True, exist_ok=True)
        out_file = f'{outpath_fig}/TESTplot_phase{phase_num}_{sub}_{ses}_{run_num}_{fnum}_{task_type}.png'
    else:
        outpath_events = f'{out_path}/TEST_gaze/Events_files_enhanced'
        Path(outpath_events).mkdir(parents=True, exist_ok=True)
        out_file = f'{outpath_events}/{sub}_{ses}_{fnum}_{task_type}_{run_num}_events.tsv'

    if not os.path.exists(out_file):
        if True:
        #try:
            run_event = pd.read_csv(row['events_path'], sep = '\t', header=0)
            run_gaze = np.load(row['gaze_path'], allow_pickle=True)['gaze2d']

            # identifies logged run start time (mri TTL 0) on clock that matches the gaze using info.player.json
            onset_time = get_onset_time(row['log_path'], row['run'], row['infoplayer_path'], run_gaze[10]['timestamp'])

            gaze_threshold = row['pupilConf_thresh'] if not pd.isna(row['pupilConf_thresh']) else 0.9
            reset_gaze_list, all_vals, clean_vals  = reset_gaze_time(run_gaze, onset_time, gaze_threshold)
            # normalized position (x and y), time (s) from onset and confidence for all gaze
            all_x, all_y, all_times, all_conf = all_vals
            all_times_arr = np.array(all_times)
            # distance from central fixation point for all gaze above confidence threshold
            clean_dist_x, clean_dist_y, clean_times, clean_conf = clean_vals

            '''
            Phase 1: plot drift-corrected gaze based on median from within trial, within ISI, within trial + ISI, and w polynomial
            '''
            plot_vals = {}
            if phase_num in [1, 2, 3, 4]:

                fix_dist_x_img, fix_dist_y_img, fix_times_img = get_fixation_gaze_things(run_event, clean_dist_x, clean_dist_y, clean_times, "image", med_fix=True)
                all_x_aligned_img, all_y_aligned_img = driftcorr_fromlast(fix_dist_x_img, fix_dist_y_img, fix_times_img, all_x, all_y, all_times)
                plot_vals['LF_img'] = {
                    'refs': ["A", "B", "C", "D"],
                    'last_fix': True,
                    'all_x_aligned': all_x_aligned_img,
                    'all_y_aligned': all_y_aligned_img,
                    'fix_times': fix_times_img,
                    'fix_dist_x': fix_dist_x_img,
                    'fix_dist_y': fix_dist_y_img,
                }

                fix_dist_x_isi, fix_dist_y_isi, fix_times_isi = get_fixation_gaze_things(run_event, clean_dist_x, clean_dist_y, clean_times, "isi", med_fix=True)
                all_x_aligned_isi, all_y_aligned_isi = driftcorr_fromlast(fix_dist_x_isi, fix_dist_y_isi, fix_times_isi, all_x, all_y, all_times)
                plot_vals['LF_isi'] = {
                    'refs': ["E", "F", "G", "H"],
                    'last_fix': True,
                    'all_x_aligned': all_x_aligned_isi,
                    'all_y_aligned': all_y_aligned_isi,
                    'fix_times': fix_times_isi,
                    'fix_dist_x': fix_dist_x_isi,
                    'fix_dist_y': fix_dist_y_isi,
                }

                fix_dist_x_img_isi, fix_dist_y_img_isi, fix_times_img_isi = get_fixation_gaze_things(run_event, clean_dist_x, clean_dist_y, clean_times, "image+isi", med_fix=True)
                all_x_aligned_img_isi, all_y_aligned_img_isi = driftcorr_fromlast(fix_dist_x_img_isi, fix_dist_y_img_isi, fix_times_img_isi, all_x, all_y, all_times)
                plot_vals['LF_img_isi'] = {
                    'refs': ["I", "J", "K", "L"],
                    'last_fix': True,
                    'all_x_aligned': all_x_aligned_img_isi,
                    'all_y_aligned': all_y_aligned_img_isi,
                    'fix_times': fix_times_img_isi,
                    'fix_dist_x': fix_dist_x_img_isi,
                    'fix_dist_y': fix_dist_y_img_isi,
                }

                deg_x = int(row['polyDeg_x']) if not pd.isna(row['polyDeg_x']) else 4
                deg_y = int(row['polyDeg_y']) if not pd.isna(row['polyDeg_y']) else 4
                anchors = [0, 1]#[0, 50]

                # distance from central fixation for high-confidence gaze captured during periods of fixation (between trials)
                fix_dist_x_pimg, fix_dist_y_pimg, fix_times_pimg = get_fixation_gaze_things(run_event, clean_dist_x, clean_dist_y, clean_times, "image")
                # fit polynomial through distance between fixation and target
                # use poly curve to apply correction to all gaze (no confidence threshold applied)
                p_of_all_x_pimg = apply_poly(fix_times_pimg, fix_dist_x_pimg, deg_x, all_times_arr, anchors=anchors)
                all_x_aligned_pimg = np.array(all_x) - (p_of_all_x_pimg)
                p_of_all_y_pimg = apply_poly(fix_times_pimg, fix_dist_y_pimg, deg_y, all_times_arr, anchors=anchors)
                all_y_aligned_pimg = np.array(all_y) - (p_of_all_y_pimg)
                plot_vals['Poly_img'] = {
                    'refs': ["M", "N", "O", "P"],
                    'last_fix': False,
                    'all_x_aligned': all_x_aligned_pimg,
                    'all_y_aligned': all_y_aligned_pimg,
                    'fix_times': fix_times_pimg,
                    'fix_dist_x': fix_dist_x_pimg,
                    'fix_dist_y': fix_dist_y_pimg,
                    'p_of_all_x': p_of_all_x_pimg,
                    'p_of_all_y': p_of_all_y_pimg,
                }

                # distance from central fixation for high-confidence gaze captured during periods of fixation (between trials)
                fix_dist_x_pisi, fix_dist_y_pisi, fix_times_pisi = get_fixation_gaze_things(run_event, clean_dist_x, clean_dist_y, clean_times, "isi")
                # fit polynomial through distance between fixation and target
                # use poly curve to apply correction to all gaze (no confidence threshold applied)
                p_of_all_x_pisi = apply_poly(fix_times_pisi, fix_dist_x_pisi, deg_x, all_times_arr, anchors=anchors)
                all_x_aligned_pisi = np.array(all_x) - (p_of_all_x_pisi)
                p_of_all_y_pisi = apply_poly(fix_times_pisi, fix_dist_y_pisi, deg_y, all_times_arr, anchors=anchors)
                all_y_aligned_pisi = np.array(all_y) - (p_of_all_y_pisi)
                plot_vals['Poly_isi'] = {
                    'refs': ["Q", "R", "S", "T"],
                    'last_fix': False,
                    'all_x_aligned': all_x_aligned_pisi,
                    'all_y_aligned': all_y_aligned_pisi,
                    'fix_times': fix_times_pisi,
                    'fix_dist_x': fix_dist_x_pisi,
                    'fix_dist_y': fix_dist_y_pisi,
                    'p_of_all_x': p_of_all_x_pisi,
                    'p_of_all_y': p_of_all_y_pisi,
                }

                # distance from central fixation for high-confidence gaze captured during periods of fixation (between trials)
                fix_dist_x_pimg_isi, fix_dist_y_pimg_isi, fix_times_pimg_isi = get_fixation_gaze_things(run_event, clean_dist_x, clean_dist_y, clean_times, "image+isi")
                # fit polynomial through distance between fixation and target
                # use poly curve to apply correction to all gaze (no confidence threshold applied)
                p_of_all_x_pimg_isi = apply_poly(fix_times_pimg_isi, fix_dist_x_pimg_isi, deg_x, all_times_arr, anchors=anchors)
                all_x_aligned_pimg_isi = np.array(all_x) - (p_of_all_x_pimg_isi)
                p_of_all_y_pimg_isi = apply_poly(fix_times_pimg_isi, fix_dist_y_pimg_isi, deg_y, all_times_arr, anchors=anchors)
                all_y_aligned_pimg_isi = np.array(all_y) - (p_of_all_y_pimg_isi)
                plot_vals['Poly_img_isi'] = {
                    'refs': ["U", "V", "W", "X"],
                    'last_fix': False,
                    'all_x_aligned': all_x_aligned_pimg_isi,
                    'all_y_aligned': all_y_aligned_pimg_isi,
                    'fix_times': fix_times_pimg_isi,
                    'fix_dist_x': fix_dist_x_pimg_isi,
                    'fix_dist_y': fix_dist_y_pimg_isi,
                    'p_of_all_x': p_of_all_x_pimg_isi,
                    'p_of_all_y': p_of_all_y_pimg_isi,
                }


            if phase_num == 1:
                # export plots to visualize the gaze drift correction for last round of QC
                mosaic = """
                    AEIMQU
                    BFJNRV
                    CGKOSW
                    DHLPTX
                """
                fs = (35, 14.0)

                fig = plt.figure(constrained_layout=True, figsize=fs)
                ax_dict = fig.subplot_mosaic(mosaic)
                run_dur = int(run_event.iloc[-1]['onset'] + 20)

                for key in plot_vals:
                    refs = plot_vals[key]['refs']
                    all_x_aligned = plot_vals[key]['all_x_aligned']
                    all_y_aligned = plot_vals[key]['all_y_aligned']
                    is_last_fix = plot_vals[key]['last_fix']
                    if not is_last_fix:
                        p_of_all_x = plot_vals[key]['p_of_all_x']
                        p_of_all_y = plot_vals[key]['p_of_all_y']
                    fix_times = plot_vals[key]['fix_times']
                    fix_dist_x = plot_vals[key]['fix_dist_x']
                    fix_dist_y = plot_vals[key]['fix_dist_y']

                    ax_dict[refs[0]].scatter(all_times, all_x, s=10, color='xkcd:light grey', alpha=all_conf)
                    ax_dict[refs[0]].scatter(all_times, all_x_aligned, c=all_conf, s=10, cmap='terrain_r', alpha=0.2)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[0]].set_ylim(-2, 2)
                    ax_dict[refs[0]].set_xlim(0, run_dur)
                    ax_dict[refs[0]].set_title(f'{sub} {ses} {run_num} {key} gaze_x')

                    ax_dict[refs[1]].scatter(clean_times, clean_dist_x, color='xkcd:light blue', s=20, alpha=0.2)
                    if is_last_fix:
                        ax_dict[refs[1]].scatter(fix_times, fix_dist_x, color='xkcd:orange', s=20, alpha=1.0)
                    else:
                        ax_dict[refs[1]].scatter(fix_times, fix_dist_x, color='xkcd:orange', s=20, alpha=0.4)
                        ax_dict[refs[1]].plot(all_times_arr, p_of_all_x, color="xkcd:black", linewidth=2)
                    ax_dict[refs[1]].set_ylim(-2, 2)
                    ax_dict[refs[1]].set_xlim(0, run_dur)
                    ax_dict[refs[1]].set_title(f'{sub} {ses} {run_num} {key} fix_distance_x')

                    ax_dict[refs[2]].scatter(all_times, all_y, color='xkcd:light grey', alpha=all_conf)
                    ax_dict[refs[2]].scatter(all_times, all_y_aligned, c=all_conf, s=10, cmap='terrain_r', alpha=0.2)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[2]].set_ylim(-2, 2)
                    ax_dict[refs[2]].set_xlim(0, run_dur)
                    ax_dict[refs[2]].set_title(f'{sub} {ses} {run_num} {key} gaze_y')

                    ax_dict[refs[3]].scatter(clean_times, clean_dist_y, color='xkcd:light blue', s=20, alpha=0.2)
                    if is_last_fix:
                        ax_dict[refs[3]].scatter(fix_times, fix_dist_y, color='xkcd:orange', s=20, alpha=1.0)
                    else:
                        ax_dict[refs[3]].scatter(fix_times, fix_dist_y, color='xkcd:orange', s=20, alpha=0.4)
                        ax_dict[refs[3]].plot(all_times_arr, p_of_all_y, color="xkcd:black", linewidth=2)
                    lb = np.min(fix_dist_y)-0.1 if np.min(fix_dist_y) < -2 else -2
                    hb = np.max(fix_dist_y)+0.1 if np.max(fix_dist_y) > 2 else 2
                    ax_dict[refs[3]].set_ylim(lb, hb)
                    ax_dict[refs[3]].set_xlim(0, run_dur)
                    ax_dict[refs[3]].set_title(f'{sub} {ses} {run_num} {key} fix_distance_y')

                fig.savefig(out_file)
                plt.close()

            elif phase_num  == 2:
                plot_vals2 = {}

                trial_dist_img, trial_x_img, trial_y_img, trial_time_img, trial_conf_img = get_trial_distances(run_event, all_x_aligned_img, all_y_aligned_img, all_times, all_conf)
                plot_vals2['LF_img'] = {
                    'refs': ["A", "B", "C", "D"],
                    'trial_dist': trial_dist_img,
                    'trial_x': trial_x_img,
                    'trial_y': trial_y_img,
                    'trial_time': trial_time_img,
                    'trial_conf': trial_conf_img,
                }

                trial_dist_isi, trial_x_isi, trial_y_isi, trial_time_isi, trial_conf_isi = get_trial_distances(run_event, all_x_aligned_isi, all_y_aligned_isi, all_times, all_conf)
                plot_vals2['LF_isi'] = {
                    'refs': ["E", "F", "G", "H"],
                    'trial_dist': trial_dist_isi,
                    'trial_x': trial_x_isi,
                    'trial_y': trial_y_isi,
                    'trial_time': trial_time_isi,
                    'trial_conf': trial_conf_isi,
                }

                trial_dist_img_isi, trial_x_img_isi, trial_y_img_isi, trial_time_img_isi, trial_conf_img_isi = get_trial_distances(run_event,
                                                                                                                                   all_x_aligned_img_isi,
                                                                                                                                   all_y_aligned_img_isi,
                                                                                                                                   all_times,
                                                                                                                                   all_conf,
                                                                                                                                   )
                plot_vals2['LF_img_isi'] = {
                    'refs': ["I", "J", "K", "L"],
                    'trial_dist': trial_dist_img_isi,
                    'trial_x': trial_x_img_isi,
                    'trial_y': trial_y_img_isi,
                    'trial_time': trial_time_img_isi,
                    'trial_conf': trial_conf_img_isi,
                }

                # TODO: generate second series of figures.
                # Within trial: plot drift_corrected gaze position over trial's timeline, for image and isi
                # Also plot other metrics: position in x and y, confidence (more blinks during ISI? more mvt in image?),
                # Plot: distribution of standard deviations during image, and during ISI
                # Plot: distance between consecutive median points of fixation
                # export plots to visualize the gaze drift correction for last round of QC
                mosaic = """
                    AEI
                    BFJ
                    CGK
                    DHL
                """
                fs = (20, 14.0)

                fig = plt.figure(constrained_layout=True, figsize=fs)
                ax_dict = fig.subplot_mosaic(mosaic)
                run_dur = int(run_event.iloc[-1]['onset'] + 20)

                for key in plot_vals2:
                    refs = plot_vals2[key]['refs']
                    trial_dist = plot_vals2[key]['trial_dist']
                    trial_x = plot_vals2[key]['trial_x']
                    trial_y = plot_vals2[key]['trial_y']
                    trial_time = plot_vals2[key]['trial_time']
                    trial_conf = plot_vals2[key]['trial_conf']

                    ax_dict[refs[0]].scatter(trial_time, trial_x, c=trial_conf, s=10, cmap='terrain_r', alpha=0.15)
                    ax_dict[refs[0]].plot([2.98, 2.98], [-0.2, 1.2], color="xkcd:red", linewidth=2)
                    ax_dict[refs[0]].set_ylim(-0.2, 1.2)
                    ax_dict[refs[0]].set_title(f'{sub} {ses} {run_num} {key} gaze_x')

                    ax_dict[refs[1]].scatter(trial_time, trial_y, c=trial_conf, s=10, cmap='terrain_r', alpha=0.15)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[1]].plot([2.98, 2.98], [-0.2, 1.2], color="xkcd:red", linewidth=2)
                    ax_dict[refs[1]].set_ylim(-0.2, 1.2)
                    ax_dict[refs[1]].set_title(f'{sub} {ses} {run_num} {key} gaze_y')

                    ax_dict[refs[2]].scatter(trial_time, trial_dist, c=trial_conf, s=10, cmap='terrain_r', alpha=0.15)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[2]].plot([2.98, 2.98], [-0.1, 7], color="xkcd:red", linewidth=2)
                    ax_dict[refs[2]].set_ylim(-0.1, 7)
                    ax_dict[refs[2]].set_title(f'{sub} {ses} {run_num} {key} gaze_dist (deg)')

                    ax_dict[refs[3]].scatter(trial_time, trial_conf, c=trial_conf, s=10, cmap='terrain_r', alpha=0.15)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[3]].plot([2.98, 2.98], [-0.1, 1.1], color="xkcd:red", linewidth=2)
                    ax_dict[refs[3]].set_ylim(-0.1, 1.1)
                    ax_dict[refs[3]].set_title(f'{sub} {ses} {run_num} {key} gaze_conf')

                fig.savefig(out_file)
                plt.close()


            elif phase_num  == 3:
                plot_vals2 = {}

                trial_dist_img, trial_x_img, trial_y_img, trial_time_img, trial_conf_img = get_trial_distances(run_event, all_x_aligned_pimg, all_y_aligned_pimg, all_times, all_conf)
                plot_vals2['Poly_img'] = {
                    'refs': ["A", "B", "C", "D"],
                    'trial_dist': trial_dist_img,
                    'trial_x': trial_x_img,
                    'trial_y': trial_y_img,
                    'trial_time': trial_time_img,
                    'trial_conf': trial_conf_img,
                }

                trial_dist_isi, trial_x_isi, trial_y_isi, trial_time_isi, trial_conf_isi = get_trial_distances(run_event, all_x_aligned_pisi, all_y_aligned_pisi, all_times, all_conf)
                plot_vals2['Poly_isi'] = {
                    'refs': ["E", "F", "G", "H"],
                    'trial_dist': trial_dist_isi,
                    'trial_x': trial_x_isi,
                    'trial_y': trial_y_isi,
                    'trial_time': trial_time_isi,
                    'trial_conf': trial_conf_isi,
                }

                trial_dist_img_isi, trial_x_img_isi, trial_y_img_isi, trial_time_img_isi, trial_conf_img_isi = get_trial_distances(run_event,
                                                                                                                                   all_x_aligned_pimg_isi,
                                                                                                                                   all_y_aligned_pimg_isi,
                                                                                                                                   all_times,
                                                                                                                                   all_conf,
                                                                                                                                   )
                plot_vals2['Poly_img_isi'] = {
                    'refs': ["I", "J", "K", "L"],
                    'trial_dist': trial_dist_img_isi,
                    'trial_x': trial_x_img_isi,
                    'trial_y': trial_y_img_isi,
                    'trial_time': trial_time_img_isi,
                    'trial_conf': trial_conf_img_isi,
                }

                # TODO: generate second series of figures.
                # Within trial: plot drift_corrected gaze position over trial's timeline, for image and isi
                # Also plot other metrics: position in x and y, confidence (more blinks during ISI? more mvt in image?),
                # Plot: distribution of standard deviations during image, and during ISI
                # Plot: distance between consecutive median points of fixation
                # export plots to visualize the gaze drift correction for last round of QC
                mosaic = """
                    AEI
                    BFJ
                    CGK
                    DHL
                """
                fs = (20, 14.0)

                fig = plt.figure(constrained_layout=True, figsize=fs)
                ax_dict = fig.subplot_mosaic(mosaic)
                run_dur = int(run_event.iloc[-1]['onset'] + 20)

                for key in plot_vals2:
                    refs = plot_vals2[key]['refs']
                    trial_dist = plot_vals2[key]['trial_dist']
                    trial_x = plot_vals2[key]['trial_x']
                    trial_y = plot_vals2[key]['trial_y']
                    trial_time = plot_vals2[key]['trial_time']
                    trial_conf = plot_vals2[key]['trial_conf']

                    ax_dict[refs[0]].scatter(trial_time, trial_x, c=trial_conf, s=10, cmap='terrain_r', alpha=0.15)
                    ax_dict[refs[0]].plot([2.98, 2.98], [-0.2, 1.2], color="xkcd:red", linewidth=2)
                    ax_dict[refs[0]].set_ylim(-0.2, 1.2)
                    ax_dict[refs[0]].set_title(f'{sub} {ses} {run_num} {key} gaze_x')

                    ax_dict[refs[1]].scatter(trial_time, trial_y, c=trial_conf, s=10, cmap='terrain_r', alpha=0.15)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[1]].plot([2.98, 2.98], [-0.2, 1.2], color="xkcd:red", linewidth=2)
                    ax_dict[refs[1]].set_ylim(-0.2, 1.2)
                    ax_dict[refs[1]].set_title(f'{sub} {ses} {run_num} {key} gaze_y')

                    ax_dict[refs[2]].scatter(trial_time, trial_dist, c=trial_conf, s=10, cmap='terrain_r', alpha=0.15)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[2]].plot([2.98, 2.98], [-0.1, 7], color="xkcd:red", linewidth=2)
                    ax_dict[refs[2]].set_ylim(-0.1, 7)
                    ax_dict[refs[2]].set_title(f'{sub} {ses} {run_num} {key} gaze_dist (deg)')

                    ax_dict[refs[3]].scatter(trial_time, trial_conf, c=trial_conf, s=10, cmap='terrain_r', alpha=0.15)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[3]].plot([2.98, 2.98], [-0.1, 1.1], color="xkcd:red", linewidth=2)
                    ax_dict[refs[3]].set_ylim(-0.1, 1.1)
                    ax_dict[refs[3]].set_title(f'{sub} {ses} {run_num} {key} gaze_conf')

                fig.savefig(out_file)
                plt.close()


            elif phase_num  == 4:

                sub_strategy = {
                    'sub-01': 'image+isi',
                    'sub-02': 'isi',
                    'sub-03': 'image',
                    'sub-06': 'isi',

                }

                fix_dist_x, fix_dist_y, fix_times = get_fixation_gaze_things(run_event, clean_dist_x, clean_dist_y, clean_times, sub_strategy[sub], med_fix=True)
                all_x_aligned, all_y_aligned = driftcorr_fromlast(fix_dist_x, fix_dist_y, fix_times, all_x, all_y, all_times)

                run_event = add_metrics_2events(
                                                run_event,
                                                all_times,
                                                all_conf,
                                                all_x,
                                                all_y,
                                                all_x_aligned,
                                                all_y_aligned,
                                                conf_thresh=gaze_threshold,
                                                )

                # export final events files w metrics on proportion of high confidence pupils per trial
                run_event.to_csv(out_file, sep='\t', header=True, index=False)

                (trial_dist, trial_x, trial_y, trial_time, trial_conf), qc_metrics = get_trial_distances_plusMetrics(
                                                                                                                     run_event,
                                                                                                                     sub_strategy[sub],
                                                                                                                     all_x_aligned,
                                                                                                                     all_y_aligned,
                                                                                                                     all_times,
                                                                                                                     all_conf,
                                                                                                                     gaze_threshold,
                                                                                                                     filter=True,
                                                                                                                     )

                fix_gz_count, fix_conf_ratio, fix_stdev_x, fix_stdev_y, pre_post_isi_dist, trial_fixCom = qc_metrics
                plot_metrics = {
                    'fix_gz_count': {
                        'refs': ['A', 'B', 'C'],
                        'metric': fix_gz_count,
                        'cmap': 'terrain_r',
                    },
                    'fix_conf_ratio': {
                        'refs': ['D', 'E', 'F'],
                        'metric': fix_conf_ratio,
                        'cmap': 'terrain_r',
                    },
                    'fix_stdev_x': {
                        'refs': ['G', 'H', 'I'],
                        'metric': fix_stdev_x,
                        'cmap': 'terrain',
                    },
                    'fix_stdev_y': {
                        'refs': ['J', 'K', 'L'],
                        'metric': fix_stdev_y,
                        'cmap': 'terrain',
                    },
                    'pre_post_isi_dist': {
                        'refs': ['M', 'N', 'O'],
                        'metric': pre_post_isi_dist,
                        'cmap': 'terrain',
                    },
                    'trial_fixCom': {
                        'refs': ['P', 'Q', 'R'],
                        'metric': trial_fixCom,
                        'cmap': 'terrain_r',
                    },
                }

                mosaic = """
                    ADGJMP
                    BEHKNQ
                    CFILOR
                """
                fs = (35, 14.0)

                fig = plt.figure(constrained_layout=True, figsize=fs)
                ax_dict = fig.subplot_mosaic(mosaic)
                run_dur = int(run_event.iloc[-1]['onset'] + 20)

                for key in plot_metrics:
                    refs = plot_metrics[key]['refs']
                    color_metric = plot_metrics[key]['metric']
                    cmap = plot_metrics[key]['cmap']

                    ax_dict[refs[0]].scatter(trial_time, trial_x, c=color_metric, s=10, cmap=cmap, alpha=0.15)
                    ax_dict[refs[0]].plot([2.98, 2.98], [-0.2, 1.2], color="xkcd:red", linewidth=2)
                    ax_dict[refs[0]].set_ylim(-0.2, 1.2)
                    ax_dict[refs[0]].set_title(f'{sub} {ses} {run_num} {sub_strategy[sub]} {key} gaze_x')

                    ax_dict[refs[1]].scatter(trial_time, trial_y, c=color_metric, s=10, cmap=cmap, alpha=0.15)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[1]].plot([2.98, 2.98], [-0.2, 1.2], color="xkcd:red", linewidth=2)
                    ax_dict[refs[1]].set_ylim(-0.2, 1.2)
                    ax_dict[refs[1]].set_title(f'{sub} {ses} {run_num} {sub_strategy[sub]} {key} gaze_y')

                    ax_dict[refs[2]].scatter(trial_time, trial_dist, c=color_metric, s=10, cmap=cmap, alpha=0.15)#'xkcd:orange', alpha=all_conf)
                    ax_dict[refs[2]].plot([2.98, 2.98], [-0.1, 7], color="xkcd:red", linewidth=2)
                    ax_dict[refs[2]].set_ylim(-0.1, 7)
                    ax_dict[refs[2]].set_title(f'{sub} {ses} {run_num} {sub_strategy[sub]} {key} gaze_dist (deg)')

                fig.savefig(out_file)
                plt.close()

                '''
                TODO: make some graphs like in phase 3 to visualize whether QC metrics reflect quality of drift correction...
                Color-code based on trial-based metrics
                '''

                '''
                # Export drift-corrected gaze, realigned timestamps, and all other metrics (pupils, etc) to bids-compliant .tsv file
                # guidelines: https://bids-specification--1128.org.readthedocs.build/en/1128/modality-specific-files/eye-tracking.html#sidecar-json-document-_eyetrackjson
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

                df_gaze = pd.DataFrame(np.array(final_gaze_list, dtype=object), columns=col_names)

                bids_out_path = f'{out_path}/TEST_gaze/final_bids_DriftCor/{sub}/{ses}'
                Path(bids_out_path).mkdir(parents=True, exist_ok=True)
                gfile_path = f'{bids_out_path}/{sub}_{ses}_{task_type}_{run_num}_eyetrack.tsv.gz'
                if os.path.exists(gfile_path):
                    # just in case session's run is done twice... note: not bids...
                    gfile_path = f'{bids_out_path}/{sub}_{ses}_{task_type}_{fnum}_{run_num}_eyetrack.tsv.gz'
                df_gaze.to_csv(gfile_path, sep='\t', header=True, index=False, compression='gzip')
                '''

        #except:
        #    print('could not process')


def main():
    '''
    This script tests different drift correction metrics for the things dataset

    In the first phase, 6 different approaches are contrasted

    In the second phase (yet to implement), additional metrics are visualized to determine whether fixation is robust during the image presentation vs ISI_0

    In the third phase (yet to implement), eye-tracking QC metrics are exported to events files,
    and bids-compliant drift corrected gaze is exported as a derivative
    '''

    args = get_arguments()
    # e.g., (elm): /unf/eyetracker/neuromod/triplets/sourcedata
    in_path = args.in_path
    out_path = args.out_path

    phase_num = args.phase_num

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
    clean_list.apply(lambda row: driftCorr_ETtests(row, out_path, phase_num), axis=1)


if __name__ == '__main__':
    sys.exit(main())
