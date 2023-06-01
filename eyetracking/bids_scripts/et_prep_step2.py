import os, glob, sys
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='clean up, label, QC and bids-formats the triplets eye tracking dataset')
    parser.add_argument('--in_path', type=str, required=True, help='absolute path to directory that contains all data (sourcedata)')
    parser.add_argument('--is_final', action='store_true', default=False, help='if true, export drift-corrected gaze into bids format')
    parser.add_argument('--run_dir', default='', type=str, help='absolute path to main code directory')
    parser.add_argument('--out_path', type=str, default='./test.tsv', help='absolute path to output file')
    args = parser.parse_args()

    return args


def compile_file_list(in_path):

    col_names = ['subject', 'session', 'run', 'task', 'file_number', 'has_pupil', 'has_gaze', 'has_eyemovie', 'has_log']
    df_files = pd.DataFrame(columns=col_names)

    # on elm, for triplets : in_path = '/unf/eyetracker/neuromod/triplets/sourcedata'
    ses_list = sorted(glob.glob(f'{in_path}/sub-*/ses-0*'))

    pupil_file_paths = []

    for ses_path in ses_list:
        [sub_num, ses_num] = ses_path.split('/')[-2:]
        events_list = sorted(glob.glob(f'{ses_path}/*task*events.tsv'))
        for event in events_list:
            ev_file = os.path.basename(event)
            [sub, ses, fnum, task_type, run_num, appendix] = ev_file.split('_')
            if sub in ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']:
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


def export_and_plot(pupil_path, in_path, out_path):
    '''
    Function accomplishes two things:
    1. export gaze and pupil metrics from .pldata (pupil's) format to .npz format
    2. compile list of gaze and pupil positions (w timestamps and confidence), and export plots for visual QCing
    '''
    sub, ses, run, task, fnum = pupil_path[1]

    task_root = out_path.split('/')[-1]
    if task_root == 'mario3':
        task = 'mario3'

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

        try:
            ev_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}_{task}_{run}_events.tsv'
            ev_lasttrial = pd.read_csv(ev_path, sep='\t', header=0).iloc[-1]
            run_dur = int(ev_lasttrial['onset'] + 20)
        except:
            print('even file did not load, using default run duration')
            run_dur = 700

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
                axes[i].set_xlim(0, run_dur)
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
    task = row["task"]

    task_root = file_path.split('/')[-1]
    if task_root == 'mario3':
        task = 'mario3'

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
    return f'{file_path}/{s}/{ses}/{s}_{ses}_{fnum}.pupil/{t}_{r}/000/info.player.json'


def get_onset_time(log_path, run_num, ip_path, gz_ts):
    onset_time_dict = {}
    TTL_0 = -1

    with open(log_path) as f:
        lines = f.readlines()
        for line in lines:
            if "fMRI TTL 0" in line:
                TTL_0 = line.split('\t')[0]
            elif "saved wide-format data to /scratch/neuromod/data" in line:
                rnum = line.split('\t')[-1].split('_')[-2]
                onset_time_dict[rnum] = float(TTL_0)
            elif "class 'src.tasks.videogame.VideoGameMultiLevel'" in line:
                rnum = line.split(': ')[-2].split('_')[-1]
                onset_time_dict[rnum] = float(TTL_0)

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


def get_fixation_gaze(df_ev, clean_dist_x, clean_dist_y, clean_times, med_fix=False, gap=0.6):
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
        trial_onset = df_ev['onset'][i]
        trial_offset = trial_onset + df_ev['duration'][i]
        fix_offset = trial_offset + df_ev['isi'][i]
        if i == 0:
            # add gaze from very first fixation at run onset (before first trial)
            trial_fd_x = []
            trial_fd_y = []
            trial_ftimes = []
            while j < len(clean_times) and clean_times[j] < (trial_onset - 0.1):
                if clean_times[j] > 3.0: # cut off first few seconds for cleaner fixations
                    trial_fd_x.append(clean_dist_x[j])
                    trial_fd_y.append(clean_dist_y[j])
                    trial_ftimes.append(clean_times[j])
                j += 1
            if len(trial_fd_x) > 0:
                med_x = np.median(trial_fd_x)
                med_y = np.median(trial_fd_y)
                trial_fd_x = np.array(trial_fd_x)
                trial_fd_y = np.array(trial_fd_y)
                trial_ftimes = np.array(trial_ftimes)
                if med_fix:
                    fix_dist_x.append(med_x)
                    fix_dist_y.append(med_y)
                    fix_times.append(trial_ftimes[0])
                else:
                    stdevx= np.std(trial_fd_x)
                    stdevy= np.std(trial_fd_y)
                    f1 = trial_fd_x < med_x + (gap*stdevx)
                    f2 = trial_fd_x > med_x - (gap*stdevx)
                    f3 = trial_fd_y < med_y + (gap*stdevy)
                    f4 = trial_fd_y > med_y - (gap*stdevy)
                    gaze_filter = f1*f2*f3*f4
                    if np.sum(gaze_filter) > 0:
                        fix_dist_x += trial_fd_x[gaze_filter].tolist()
                        fix_dist_y += trial_fd_y[gaze_filter].tolist()
                        fix_times += trial_ftimes[gaze_filter].tolist()

        # add gaze from post-trial fixation period
        trial_fd_x = []
        trial_fd_y = []
        trial_ftimes = []
        while j < len(clean_times) and clean_times[j] < fix_offset:
            # + 0.8 = 800ms (0.8s) after trial offset to account for saccade
            if clean_times[j] > (trial_offset + 0.8) and clean_times[j] < (fix_offset - 0.1):
                trial_fd_x.append(clean_dist_x[j])
                trial_fd_y.append(clean_dist_y[j])
                trial_ftimes.append(clean_times[j])
            j += 1
        if len(trial_fd_x) > 0:
            med_x = np.median(trial_fd_x)
            med_y = np.median(trial_fd_y)
            trial_fd_x = np.array(trial_fd_x)
            trial_fd_y = np.array(trial_fd_y)
            trial_ftimes = np.array(trial_ftimes)
            if med_fix:
                fix_dist_x.append(med_x)
                fix_dist_y.append(med_y)
                fix_times.append(trial_ftimes[0])
            else:
                stdevx= np.std(trial_fd_x)
                stdevy= np.std(trial_fd_y)
                f1 = trial_fd_x < med_x + (gap*stdevx)
                f2 = trial_fd_x > med_x - (gap*stdevx)
                f3 = trial_fd_y < med_y + (gap*stdevy)
                f4 = trial_fd_y > med_y - (gap*stdevy)
                gaze_filter = f1*f2*f3*f4
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


def assign_Compliance2trial(df_ev, vals_times, vals_x, vals_y, task):

    fixCompliance_per_trials = {}
    j = 0

    for i in range(df_ev.shape[0]):
        trial_number = df_ev['TrialNumber'][i]

        if task == 'task-thingsmemory':
            fixation_onset = df_ev['onset'][i]
            fixation_offset = trial_onset + df_ev['duration'][i]
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
                x_comp = abs(vals_x[j] - 0.5) < (1/17.5)
                y_comp = abs(vals_y[j] - 0.5) < (1/14.0)
                trial_comp.append(x_comp and y_comp)
            j += 1

        num_gaze = len(trial_comp)
        if task == 'task-thingsmemory':
            assert gaze_count == num_gaze
        fixCompliance_per_trials[trial_number] = np.sum(trial_comp)/num_gaze if num_gaze > 0 else np.nan

    df_ev['fixation_compliance_ratio'] = df_ev.apply(lambda row: fixCompliance_per_trials[row['TrialNumber']], axis=1)

    return df_ev


def assign_gzMetrics2trial_mario(df_ev, vals_times, vals_conf, vals_x, vals_y, conf_thresh=0.9):

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
    fix_compliance = []
    gaze_per_trial = []

    j = 0
    for i in range(df_ev.shape[0]):
        trial_onset = df_ev['onset'][i]
        trial_type = df_ev['trial_type'][i]

        if trial_type == 'fixation_dot':
            trial_comp = []
            trial_offset = trial_onset + df_ev['duration'][i]

            while j < len(vals_times) and vals_times[j] < trial_offset:
                if vals_times[j] > trial_onset:
                    # is gaze within 1 degree of visual angle of central fixation in x and y?
                    x_comp = abs(vals_x[j] - 0.5) < (1/17.5)
                    y_comp = abs(vals_y[j] - 0.5) < (1/14.0)
                    trial_comp.append(x_comp and y_comp)
                j += 1
            num_gaze = len(trial_comp)
            if num_gaze > 0:
                fix_compliance.append(np.sum(trial_comp)/num_gaze)
                gaze_per_trial.append(num_gaze)
            else:
                fix_compliance.append(np.nan)
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

            num_gaze = len(trial_confs)
            if num_gaze > 0:
                gaze_confidence.append(np.sum(trial_conf)/num_gaze)
                gaze_per_trial.append(num_gaze)
            else:
                gaze_confidence.append(np.nan)
                gaze_per_trial.append(np.nan)
            fix_compliance.append(np.nan)

        else:
            gaze_confidence.append(np.nan)
            gaze_per_trial.append(np.nan)
            fix_compliance.append(np.nan)

    # TODO:  insert 3 arrays as new columns in df_ev
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


def median_clean(gz_times, dist_x, dist_y):
    '''
    Within bins of 1/100 the number of gaze,
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
                        filtered_times.append(gz_times[i])
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


def driftCorr_EToutput(row, out_path, is_final=False):

    [sub, ses, fnum, task_type, run_num, appendix] = os.path.basename(row['events_path']).split('_')

    task_root = out_path.split('/')[-1]
    pseudo_task = 'task-mario3' if task_root == 'mario3' else task_type
    print(sub, ses, fnum, pseudo_task, run_num)

    if is_final:
        outpath_events = f'{out_path}/Events_files'
        Path(outpath_events).mkdir(parents=True, exist_ok=True)
        out_file = f'{outpath_events}/{sub}_{ses}_{fnum}_{pseudo_task}_{run_num}_events.tsv'
    else:
        outpath_fig = os.path.join(out_path, 'DC_gaze')
        Path(outpath_fig).mkdir(parents=True, exist_ok=True)
        out_file = f'{out_path}/DC_gaze/{sub}_{ses}_{run_num}_{fnum}_{pseudo_task}_DCplot.png'

    if not os.path.exists(out_file):
        try:
            run_event = pd.read_csv(row['events_path'], sep = '\t', header=0)
            run_gaze = np.load(row['gaze_path'], allow_pickle=True)['gaze2d']

            # identifies logged run start time (mri TTL 0) on clock that matches the gaze using info.player.json
            onset_time = get_onset_time(row['log_path'], row['run'], row['infoplayer_path'], run_gaze[0]['timestamp'])

            gaze_threshold = row['pupilConf_thresh'] if not pd.isna(row['pupilConf_thresh']) else 0.9
            reset_gaze_list, all_vals, clean_vals  = reset_gaze_time(run_gaze, onset_time, gaze_threshold)
            # normalized position (x and y), time (s) from onset and confidence for all gaze
            all_x, all_y, all_times, all_conf = all_vals
            all_times_arr = np.array(all_times)
            # distance from central fixation point for all gaze above confidence threshold
            clean_dist_x, clean_dist_y, clean_times, clean_conf = clean_vals

            if row['use_latestFix']==1.0:
                '''
                use latest point of fixation to realign the gaze
                '''
                #TODO: adjust get_fixation_gaze and driftcorr_fromlast to other datasets
                fix_dist_x, fix_dist_y, fix_times = get_fixation_gaze(run_event, clean_dist_x, clean_dist_y, clean_times, med_fix=True)
                all_x_aligned, all_y_aligned = driftcorr_fromlast(fix_dist_x, fix_dist_y, fix_times, all_x, all_y, all_times)
            else:
                # distance from central fixation for high-confidence gaze captured during periods of fixation (between trials)
                fix_dist_x, fix_dist_y, fix_times = get_fixation_gaze(run_event, clean_dist_x, clean_dist_y, clean_times)

                deg_x = int(row['polyDeg_x']) if not pd.isna(row['polyDeg_x']) else 4
                deg_y = int(row['polyDeg_y']) if not pd.isna(row['polyDeg_y']) else 4
                anchors = [0, 50]
                # fit polynomial through distance between fixation and target
                # and use it apply correction to all gaze (no confidence threshold applied)
                p_of_all_x = apply_poly(fix_times, fix_dist_x, deg_x, all_times_arr, anchors=anchors)
                all_x_aligned = np.array(all_x) - (p_of_all_x)

                p_of_all_y = apply_poly(fix_times, fix_dist_y, deg_y, all_times_arr, anchors=anchors)
                all_y_aligned = np.array(all_y) - (p_of_all_y)

            if 'mario' in pseudo_task:
                run_event = assign_gzMetrics2trial_mario(run_event, all_times, all_conf, all_x_aligned, all_y_aligned, conf_thresh=0.9)
            else:
                # for each trial, derive % of above-threshold gaze and add metric to events file
                run_event = assign_gazeConf2trial(run_event, all_times, all_conf, task_type, conf_thresh=0.9)
                if gaze_threshold != 0.9:
                    run_event = assign_gazeConf2trial(run_event, all_times, all_conf, task_type, conf_thresh=gaze_threshold, add_count=False)
                # Measure fixation compliance during trial (THINGS) or during preceeding fixation
                run_event = assign_Compliance2trial(run_event, all_times, all_x_aligned, all_y_aligned, task_type)

            if is_final:
                # export final events files w metrics on proportion of high confidence pupils per trial
                run_event.to_csv(out_file, sep='\t', header=True, index=False)

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

                bids_out_path = f'{out_path}/final_bids/{sub}/{ses}'
                Path(bids_out_path).mkdir(parents=True, exist_ok=True)
                gfile_path = f'{bids_out_path}/{sub}_{ses}_{pseudo_task}_{run_num}_eyetrack.tsv.gz'
                if os.path.exists(gfile_path):
                    # just in case session's run is done twice... note: not bids...
                    gfile_path = f'{bids_out_path}/{sub}_{ses}_{pseudo_task}_{fnum}_{run_num}_eyetrack.tsv.gz'
                df_gaze.to_csv(gfile_path, sep='\t', header=True, index=False, compression='gzip')

            else:
                # export plots to visulize the gaze drift correction for last round of QC
                fig, axes = plt.subplots(5, 1, figsize=(7, 17.5))
                run_dur = int(run_event.iloc[-1]['onset'] + 20)

                axes[0].scatter(all_times, all_x, color='xkcd:blue', alpha=all_conf)
                axes[0].scatter(all_times, all_x_aligned, color='xkcd:orange', alpha=all_conf)
                axes[0].set_ylim(-2, 2)
                axes[0].set_xlim(0, run_dur)
                axes[0].set_title(f'{sub} {task_type} {ses} {run_num} gaze_x')

                axes[1].scatter(all_times, all_y, color='xkcd:blue', alpha=all_conf)
                axes[1].scatter(all_times, all_y_aligned, color='xkcd:orange', alpha=all_conf)
                axes[1].set_ylim(-2, 2)
                axes[1].set_xlim(0, run_dur)
                axes[1].set_title(f'{sub} {task_type} {ses} {run_num} gaze_y')

                if row['use_latestFix']==1.0:
                    axes[2].scatter(clean_times, clean_dist_x, color='xkcd:blue', s=20, alpha=0.4)
                    axes[2].scatter(fix_times, fix_dist_x, color='xkcd:orange', s=20, alpha=1.0)
                else:
                    axes[2].scatter(fix_times, fix_dist_x, color='xkcd:blue', s=20, alpha=0.4)
                    #axes[2].scatter(mf_fix_times, mf_fix_dist_x, s=20, alpha=0.4)
                    axes[2].plot(all_times_arr, p_of_all_x, color="xkcd:red", linewidth=2)
                axes[2].set_ylim(-2, 2)
                axes[2].set_xlim(0, run_dur)
                axes[2].set_title(f'{sub} {task_type} {ses} {run_num} fix_distance_x')

                if row['use_latestFix']==1.0:
                    axes[3].scatter(clean_times, clean_dist_y, color='xkcd:blue', s=20, alpha=0.4)
                    axes[3].scatter(fix_times, fix_dist_y, color='xkcd:orange', s=20, alpha=1.0)
                else:
                    axes[3].scatter(fix_times, fix_dist_y, color='xkcd:orange', s=20, alpha=0.4)
                    #axes[3].scatter(mf_fix_times, mf_fix_dist_y, s=20, alpha=0.4)
                    axes[3].plot(all_times_arr, p_of_all_y, color="xkcd:red", linewidth=2)
                lb = np.min(fix_dist_y)-0.1 if np.min(fix_dist_y) < -2 else -2
                hb = np.max(fix_dist_y)+0.1 if np.max(fix_dist_y) > 2 else 2
                axes[3].set_ylim(lb, hb)
                axes[3].set_xlim(0, run_dur)
                axes[3].set_title(f'{sub} {task_type} {ses} {run_num} fix_distance_y')

                axes[4].scatter(run_event['onset'].to_numpy()+2.0, run_event[f'gaze_confidence_ratio_cThresh{gaze_threshold}'].to_numpy())
                axes[4].set_ylim(-0.1, 1.1)
                axes[4].set_xlim(0, run_dur)
                axes[4].set_title(f'{sub} {task_type} {ses} {run_num} ratio >{str(gaze_threshold)} confidence per trial')

                fig.savefig(out_file)
                plt.close()
        except:
            print('could not process')


def main():
    '''
    This ccript applies drift correction to gaze based on known periods of fixations (ground truth).
    The default approach is to draw a polynomial of deg=4 (in x and y) through the distance between the mapped gaze
    and known target positions during periods of fixations plotted over time (throughout a run's duration).

    For each run, options include: lowering the pupil confidence threshold for gaze included in the drift correction,
    specifying the degrees of the polynomial in x and y, and performing drift correction based strictly on the previous fixation,
    ignoring all others (e.g., in cases where sudden head motion gives a poor fit to a polynomial)

    "is_final" is False:
        the script exports charts plotting the drift corrected gaze over time to help QC the corrected gaze data

    "is_final" is True:
        the script exports bids-compliant gaze and pupil metrics in .tsv.gz format,
        according to the following proposed bids extension guidelines:
        https://bids-specification--1128.org.readthedocs.build/en/1128/modality-specific-files/eye-tracking.html#sidecar-json-document-_eyetrackjson
    '''

    args = get_arguments()
    # e.g., (elm): /unf/eyetracker/neuromod/triplets/sourcedata
    in_path = args.in_path
    out_path = args.out_path

    is_final = args.is_final

    # load list of valid files (those deserialized and exported as npz in step 2 that passed QC)
    outpath_report = os.path.join(out_path, 'QC_gaze')
    Path(outpath_report).mkdir(parents=True, exist_ok=True)

    if is_final:
        file_list = pd.read_csv(f'{outpath_report}/QCed_finalbids_list.tsv', sep='\t', header=0)
    else:
        file_list = pd.read_csv(f'{outpath_report}/QCed_file_list.tsv', sep='\t', header=0)

    clean_list = file_list[file_list['DO_NOT_USE']!=1.0]
    if is_final:
        final_list = clean_list[clean_list['Fails_DriftCorr']!=1.0]
        clean_list = final_list
    clean_list['gaze_path'] = clean_list.apply(lambda row: create_gaze_path(row, out_path), axis=1)
    clean_list['events_path'] = clean_list.apply(lambda row: create_event_path(row, in_path), axis=1)
    clean_list['log_path'] = clean_list.apply(lambda row: create_event_path(row, in_path, log=True), axis=1)
    clean_list['infoplayer_path'] = clean_list.apply(lambda row: create_ip_path(row, in_path), axis=1)

    # implement drift correction on each run
    clean_list.apply(lambda row: driftCorr_EToutput(row, out_path, is_final), axis=1)


if __name__ == '__main__':
    sys.exit(main())
