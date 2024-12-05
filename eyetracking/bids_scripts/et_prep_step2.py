import os, glob, sys, json

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

from friends_driftCor import driftCorr_ETfriends
from things_driftCor import driftCorr_ETthings
from utils import get_list, add_file_paths, driftcorr_fromlast, get_onset_time
from utils import reset_gaze_time, format_gaze, apply_poly


def get_arguments():
    parser = argparse.ArgumentParser(
        description="cleans up, labels, QCs and bids-formats eye tracking datasets"
    )
    parser.add_argument(
        '--in_path',
        type=str,
        required=True,
        help='absolute path to directory that contains all data (sourcedata)'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['things', 'emotionsvideos', 'mario3', 'mariostars', 'triplets',
                 'floc', 'retino', 'friends'],
        # not included (no fixation): mario
        help='task to analyse.'
    )
    parser.add_argument(
        '--is_final',
        action='store_true',
        default=False,
        help='if true, export drift-corrected gaze into bids format'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./test.tsv',
        help='absolute path to analysis directory'
    )
    parser.add_argument(
        '--mkv_path',
        type=str,
        default='.',
        help='absolute path to stimulus directory (friends task only)'
    )

    return parser.parse_args()


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


def get_fixation_gaze(
    df_ev: pd.DataFrame,
    clean_dist_x: list,
    clean_dist_y: list,
    clean_times: list,
    task: str,
    med_fix: bool=False,
    gap: float=0.6,
) -> tuple:
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
            if 'mario' in task:
                fixation_onset = df_ev['onset'][i]
                fixation_offset = fixation_onset + df_ev['duration'][i]

            elif task == 'task-emotionvideos':
                fixation_onset = df_ev['onset_fixation_flip'][i]
                fixation_offset = df_ev['onset_video_flip'][i]

            elif task in ['task-wordsfamiliarity', 'task-triplets']:
                fixation_onset = df_ev['onset'][i] - 3.0 if i == 0 else df_ev['onset'][i-1] + df_ev['duration'][i-1]
                fixation_offset = df_ev['onset'][i]

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


def get_interfix_dist(df_ev, clean_times, clean_dist_x, clean_dist_y, task):
    fix_dict = {}
    j = 0
    fix_idx = []

    for i in range(df_ev.shape[0]):
        has_fixation = True
        if 'mario' in task:
            if df_ev['trial_type'][i] != 'fixation_dot':
                has_fixation = False

        if has_fixation:
            if 'mario' in task:
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
            while j < len(clean_times) and clean_times[j] < fixation_offset:
                # + 0.8 = 800ms (0.8s) after trial offset to account for saccade
                if clean_times[j] > (fixation_onset + 0.8) and clean_times[j] < (fixation_offset - 0.1):
                    trial_fd_x.append(clean_dist_x[j])
                    trial_fd_y.append(clean_dist_y[j])
                j += 1

            if len(trial_fd_x) > 0:
                fix_dict[i] = {
                    'med_x': np.median(trial_fd_x),
                    'med_y': np.median(trial_fd_y),
                }
                fix_idx.append(i)

    dist_2prev = []
    dist_2next = []
    dist_in_pix = 4164 # in pixels
    k = 0
    for i in range(df_ev.shape[0]):
        if i in fix_dict:
            curr_x = fix_dict[i]['med_x']*1280
            curr_y = fix_dict[i]['med_y']*1024
            if k > 0:
                prev_x = fix_dict[fix_idx[k-1]]['med_x']*1280
                prev_y = fix_dict[fix_idx[k-1]]['med_y']*1024
                pre_vectors = np.array([[prev_x, prev_y, dist_in_pix], [curr_x, curr_y, dist_in_pix]])
                pre_distance = np.rad2deg(np.arccos(1.0 - pdist(pre_vectors, metric='cosine')))[0]
                dist_2prev.append(pre_distance)
            else:
                dist_2prev.append(np.nan)

            if k+1 < len(fix_idx):
                post_x = fix_dict[fix_idx[k+1]]['med_x']*1280
                post_y = fix_dict[fix_idx[k+1]]['med_y']*1024
                post_vectors = np.array([[curr_x, curr_y, dist_in_pix], [post_x, post_y, dist_in_pix]])
                post_distance = np.rad2deg(np.arccos(1.0 - pdist(post_vectors, metric='cosine')))[0]
                dist_2next.append(post_distance)
            else:
                dist_2next.append(np.nan)
            k += 1

        else:
            dist_2prev.append(np.nan)
            dist_2next.append(np.nan)

    df_ev.insert(loc=df_ev.shape[1], column='fix_dist2prev',
                 value=dist_2prev, allow_duplicates=True)
    df_ev.insert(loc=df_ev.shape[1], column='fix_dist2next',
                 value=dist_2next, allow_duplicates=True)

    return df_ev


def poly_driftcorr(
    row,
    task_root,
    clean_dist_x,
    clean_dist_y,
    clean_times,
    all_times,
    all_x,
    all_y,
) -> tuple:

    deg_x = int(row['polyDeg_x']) if not pd.isna(row['polyDeg_x']) else 4
    deg_y = int(row['polyDeg_y']) if not pd.isna(row['polyDeg_y']) else 4
    anchors = [0, 1]#[0, 50]

    # remove 3-9s of gaze data at begining and end for stability
    otime = 6 if task_root == 'floc' else 3
    fix_dist_x = clean_dist_x[250*otime:-(250*9)]
    fix_dist_y = clean_dist_y[250*otime:-(250*9)]
    fix_times = clean_times[250*otime:-(250*9)]

    # remove fixation points > 0.15 (normalized screen) from polynomial to remove outliers
    p_of_fix_x = apply_poly(fix_times, fix_dist_x, deg_x, np.array(fix_times), anchors=anchors)
    p_of_fix_y = apply_poly(fix_times, fix_dist_y, deg_y, np.array(fix_times), anchors=anchors)

    x_filter = np.absolute(np.array(fix_dist_x) - p_of_fix_x) < 0.15
    y_filter = np.absolute(np.array(fix_dist_y) - p_of_fix_y) < 0.15
    fix_filter = (x_filter * y_filter).astype(bool)

    fix_dist_x = fix_dist_x[fix_filter]
    fix_dist_y = fix_dist_y[fix_filter]
    fix_times = fix_times[fix_filter]

    # fit polynomial through distance between fixation and target
    # use poly curve to apply correction to all gaze (no confidence threshold applied)
    p_of_all_x = apply_poly(fix_times, fix_dist_x, deg_x, np.array(all_times), anchors=anchors)
    all_x_aligned = np.array(all_x) - (p_of_all_x)

    p_of_all_y = apply_poly(fix_times, fix_dist_y, deg_y, np.array(all_times), anchors=anchors)
    all_y_aligned = np.array(all_y) - (p_of_all_y)

    return (fix_dist_x, fix_dist_y, fix_times, all_x_aligned, all_y_aligned)


def make_QC_figure(
    sub: str,
    ses: str,
    task: str,
    run_num: str,
    run_event: pd.DataFrame,
    clean_times: list,
    clean_dist_x: list,
    clean_dist_y: list,
    fix_times: list,
    fix_dist_x: list,
    fix_dist_y: list,
    all_times: list,
    all_x: list,
    all_x_aligned: list,
    all_y: list,
    all_y_aligned: list,
    out_fig_path,
) -> None:
    """
    Export figures to assess drift correction per run
    """
    mosaic = """
        AB
        CD
    """
    fs = (15, 7.0)

    fig = plt.figure(constrained_layout=True, figsize=fs)
    ax_dict = fig.subplot_mosaic(mosaic)
    run_dur = int(run_event.iloc[-1]['onset'] + 20)

    ax_dict["A"].scatter(all_times, all_x, s=10, color='xkcd:light grey', alpha=all_conf)
    ax_dict["A"].scatter(all_times, all_x_aligned, c=all_conf, s=10, cmap='terrain_r', alpha=0.2)
    ax_dict["A"].set_ylim(-2, 2)
    ax_dict["A"].set_xlim(0, run_dur)
    ax_dict["A"].set_title(f'{sub} {task} {ses} {run_num} gaze_x')

    ax_dict["B"].scatter(all_times, all_y, color='xkcd:light grey', alpha=all_conf)
    ax_dict["B"].scatter(all_times, all_y_aligned, c=all_conf, s=10, cmap='terrain_r', alpha=0.2)
    ax_dict["B"].set_ylim(-2, 2)
    ax_dict["B"].set_xlim(0, run_dur)
    ax_dict["B"].set_title(f'{sub} {task} {ses} {run_num} gaze_y')

    ax_dict["C"].scatter(clean_times, clean_dist_x, color='xkcd:light blue', s=20, alpha=0.2)
    ax_dict["C"].scatter(fix_times, fix_dist_x, color='xkcd:orange', s=20, alpha=1.0)
    ax_dict["C"].set_ylim(-2, 2)
    ax_dict["C"].set_xlim(0, run_dur)
    ax_dict["C"].set_title(f'{sub} {task} {ses} {run_num} fix_distance_x')

    ax_dict["D"].scatter(clean_times, clean_dist_y, color='xkcd:light blue', s=20, alpha=0.2)
    ax_dict["D"].scatter(fix_times, fix_dist_y, color='xkcd:orange', s=20, alpha=1.0)
    lb = np.min(fix_dist_y)-0.1 if np.min(fix_dist_y) < -2 else -2
    hb = np.max(fix_dist_y)+0.1 if np.max(fix_dist_y) > 2 else 2
    ax_dict["D"].set_ylim(lb, hb)
    ax_dict["D"].set_xlim(0, run_dur)
    ax_dict["D"].set_title(f'{sub} {task} {ses} {run_num} fix_distance_y')

    fig.savefig(out_fig_path)
    plt.close()


def driftCorr_EToutput(
    row,
    out_path,
    is_final=False,
) -> None:

    task_root = out_path.split('/')[-1]

    if task_root in ['retino', 'floc']:
        skip_run_num = True
        [sub, ses, fnum, task_type, appendix] = os.path.basename(
            row['events_path']
        ).split('_')
        run_num = run2task_mapping[task_root][task_type]
    else:
        skip_run_num = False
        [sub, ses, fnum, task_type, run_num, appendix] = os.path.basename(
            row['events_path']
        ).split('_')

    pseudo_task = 'task-mario3' if task_root == 'mario3' else task_type
    print(sub, ses, fnum, pseudo_task, run_num)

    if is_final:
        outpath_events = f'{out_path}/Events_files_enhanced'
        Path(outpath_events).mkdir(parents=True, exist_ok=True)
        out_file = f'{outpath_events}/{sub}_{ses}_{fnum}_{pseudo_task}_{run_num}_events.tsv'
    else:
        outpath_fig = os.path.join(out_path, 'DC_gaze')
        Path(outpath_fig).mkdir(parents=True, exist_ok=True)
        out_file = f'{out_path}/DC_gaze/{sub}_{ses}_{run_num}_{fnum}_{pseudo_task}_DCplot.png'

    if not os.path.exists(out_file):
        #if True:
        try:
            run_event = pd.read_csv(row['events_path'], sep = '\t', header=0)
            run_gaze = np.load(row['gaze_path'], allow_pickle=True)['gaze2d']

            '''
            identifies logged run start time (mri TTL 0) on clock that matches
            the gaze using info.player.json
            '''
            onset_time = get_onset_time(
                row['log_path'],
                row['run'],
                row['infoplayer_path'],
                run_gaze[10]['timestamp'],
            )

            '''
            Realign gaze timestamps with run onset, and filter out
            below-threshold gaze ("clean")
            '''
            gaze_threshold = row['pupilConf_thresh'] if not pd.isna(row['pupilConf_thresh']) else 0.9
            reset_gaze_list, all_vals, clean_vals  = reset_gaze_time(run_gaze, onset_time, gaze_threshold)
            # normalized position (x and y), time (s) from onset and confidence for all gaze
            all_x, all_y, all_times, all_conf = all_vals
            # distance from central fixation point for all gaze above confidence threshold
            clean_dist_x, clean_dist_y, clean_times, clean_conf = clean_vals

            """
            Perform drift correction
            """
            if task_root in ['retino', 'floc']:
                """
                Use polynomial to drift correct, since no discrete periods of
                fixation (tasks require continuous fixation, so all gaze are
                fixations).
                Default degrees are 4 in x and in y. They can be
                adjusted for each run (range [1-4]) by specifying them in
                'polyDeg_x' and 'polyDeg_y' columns.
                """
                (
                    fix_dist_x,
                    fix_dist_y,
                    fix_times,
                    all_x_aligned,
                    all_y_aligned,
                ) = poly_driftcorr(
                    row,
                    task_root,
                    clean_dist_x,
                    clean_dist_y,
                    clean_times,
                    all_times,
                    all_x,
                    all_y,
                )
            else:
                '''
                Use median gaze from latest point of central fixation to
                realign each trial's the gaze
                (for all tasks beside retino, fLoc and things)
                '''
                fix_dist_x, fix_dist_y, fix_times = get_fixation_gaze(
                    run_event,
                    clean_dist_x,
                    clean_dist_y,
                    clean_times,
                    pseudo_task,
                    med_fix=True,
                )
                all_x_aligned, all_y_aligned = driftcorr_fromlast(
                    fix_dist_x,
                    fix_dist_y,
                    fix_times,
                    all_x,
                    all_y,
                    all_times,
                )

            if is_final:
                """
                export final events files w added metrics on eyetracking &
                fixation quality per trial
                """
                if task_root not in ['retino', 'floc']:
                    run_event = get_interfix_dist(run_event, clean_times, clean_dist_x, clean_dist_y, pseudo_task)
                    run_event.to_csv(out_file, sep='\t', header=True, index=False)

                """
                Export drift-corrected gaze, realigned timestamps,
                and all other metrics (pupils, etc) to bids-compliant .tsv file.
                Guidelines: https://bids-specification--1128.org.readthedocs.build/en/1128/modality-specific-files/eye-tracking.html#sidecar-json-document-_eyetrackjson
                """
                df_gaze = format_gaze(
                    all_x_aligned,
                    all_y_aligned,
                    all_times,
                    reset_gaze_list,
                )

                bids_out_path = f'{out_path}/final_bids_DriftCor/{sub}/{ses}'
                Path(bids_out_path).mkdir(parents=True, exist_ok=True)
                gfile_path = f'{bids_out_path}/{sub}_{ses}_{pseudo_task}_{run_num}_eyetrack.tsv.gz'
                if os.path.exists(gfile_path):
                    # just in case session's run is done twice... note: not bids...
                    gfile_path = f'{bids_out_path}/{sub}_{ses}_{pseudo_task}_{fnum}_{run_num}_eyetrack.tsv.gz'
                df_gaze.to_csv(gfile_path, sep='\t', header=True, index=False, compression='gzip')

            else:
                '''
                plot QC figures (one per run) to assess drift correction
                '''
                make_QC_figure(
                    sub,
                    ses,
                    pseudo_task,
                    run_num,
                    run_event,
                    clean_times,
                    clean_dist_x,
                    clean_dist_y,
                    fix_times,
                    fix_dist_x,
                    fix_dist_y,
                    all_times,
                    all_x,
                    all_x_aligned,
                    all_y,
                    all_y_aligned,
                    out_file,
                )

        except:
            print('could not process')


def main():
    '''
    This script applies drift correction to gaze. For most tasks, it corrects
    drift based on known periods of fixation. The default approach is to realign
    gaze with the median gaze position during the central fixation period that
    precedes each trial.

    For each run, the pupil confidence threshold for gaze included in the drift
    correction can be adjusted.

    "is_final" is False:
        the script exports charts plotting the drift corrected gaze over time
        to QC the gaze realignment

    "is_final" is True:
        the script exports bids-compliant gaze and pupil metrics in .tsv.gz
        format, according to the following proposed bids extension guidelines:
        https://bids-specification--1128.org.readthedocs.build/en/1128/modality-specific-files/eye-tracking.html#sidecar-json-document-_eyetrackjson
    '''

    args = get_arguments()
    # e.g., (elm): /unf/eyetracker/neuromod/triplets/sourcedata
    in_path = args.in_path
    out_path = args.out_path
    mkv_path = args.mkv_path if args.task == 'friends' else None
    is_final = args.is_final

    # load list of valid files (those deserialized and exported as npz in step 2 that passed QC)
    clean_list = get_list(out_path, is_final)
    clean_list = add_file_paths(
        clean_list,
        in_path,
        out_path,
        mkv_path,
        args.task,
    )

    # implement drift correction on each run
    if args.task == 'things':
        clean_list.apply(lambda row: driftCorr_ETthings(row, out_path, is_final), axis=1)
    elif args.task == 'friends':
        clean_list.apply(lambda row: driftCorr_ETfriends(row, out_path, False, is_final), axis=1)
    else:
        clean_list.apply(lambda row: driftCorr_EToutput(row, out_path, is_final), axis=1)


if __name__ == '__main__':
    sys.exit(main())
