import os, sys

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

from utils import get_list, add_file_paths, apply_poly, apply_gaussian
from utils import reset_gaze_time, format_gaze


def get_arguments():
    parser = argparse.ArgumentParser(
        description='clean up, label, QC and bids-formats the friends'
        'eye-tracking dataset'
    )
    parser.add_argument(
        '--in_path',
        type=str,
        required=True,
        help='absolute path to directory that contains all data (sourcedata)',
    )
    parser.add_argument(
        '--is_final',
        action='store_true',
        default=False,
        help='if true, export drift-corrected gaze into bids format',
    )
    parser.add_argument(
        '--use_poly',
        action='store_true',
        default=False,
        help="if true, use a polynomial to fit the distance between the"
        "measured gaze and DeepGaze's estimate to model drift, else use a"
        "gaussian filter within a sliding window (default).",
    )
    parser.add_argument(
        '--out_path',
        type=str,
        required=True,
        help='absolute path to analysis directory'
    )
    parser.add_argument(
        '--mkv_path',
        type=str,
        required=True,
        help='absolute path to stimulus directory',
    )

    return parser.parse_args()


def get_indices(
    film_path: str,
    gz: np.array,
):
    '''
    Determine onset time for first movie frame based on last gaze timestamp
    There are no logged times for movie frames for most runs (seasons 5 and 6)

    The movie and eyetracking camera onsets are not perfectly aligned
    (the movie lags by ~40ms), but their offsets are.
    This approach works backward based on the movie frame count and fps rate to
    estimate the movie onset, in order to align movie frames with reccorded gaze
    '''

    '''
    Count the number of frames in episode .mkv
    This approach is MUCH slower but more accurate than
    cap.get(cv2.CAP_PROP_FRAME_COUNT), which is essential for the frame count to
    match the Deepgaze framewise output
    '''
    #print(film_path)
    cap = cv2.VideoCapture(film_path)
    success = True
    frame_count = 0
    while success:
        success, image = cap.read()
        if success:
            frame_count += 1

    '''
    Determine onset time for first movie frame based on gaze timestamps
    '''
    fps = cap.get(cv2.CAP_PROP_FPS)
    lag = gz[-1]['timestamp'] - gz[0]['timestamp'] - frame_count/fps
    interval = lag - 1/fps

    zero_idx = 0
    while gz[zero_idx]['timestamp'] - gz[0]['timestamp'] < interval:
        zero_idx += 1

    if lag < 0:
        print('onset time estimated from gaze timestamps')
        zero_idx = 109

    return frame_count, zero_idx, gz[zero_idx]['timestamp'], fps


def gaze_2_frame(frame_count, fps, x_vals, y_vals, time_vals):
    '''
    Assign gaze positions to movie frames
    Input:
        x_vals (list of float): x coordinates
        y_vals (list of float): y coordinates
        time_vals (list of float): gaze's timestamps
    Output:
        gaze_perFrame (numpy array of ragged lists): indexable array of gaze values for each frame
    '''
    frame_dur = 1/fps
    gaze_perFrame = np.empty(frame_count, dtype='object')

    frame_vals = []
    i = 0

    for frame_num in range(frame_count):
        while i < len(time_vals) and time_vals[i] < (frame_num + 1) * frame_dur:
            frame_vals.append([time_vals[i], x_vals[i], y_vals[i]])
            i += 1
        gaze_perFrame[frame_num] = frame_vals
        frame_vals = []

    return gaze_perFrame


def get_distances(
    frame_count: int,
    gazes_perframe: np.array,
    DGvals_perframe: np.array,
    use_deepgaze: bool=False,
    x_mass: float=0.5,
    y_mass: float=0.605,
) -> tuple:
    '''
    Calculates distances between mean gaze (averaged per frame) and DeepGaze
    prediction.
    If use_deepgaze is False, then calculates distances to target centers
    of mass x = 0.5 and y = 0.605
    '''
    frame_nums = []

    et_x = []
    et_y = []

    dist_x = []
    dist_y = []

    dg_x = []
    dg_y = []

    for i in range(frame_count):
        if len(gazes_perframe[i]) > 0:
            x_et, y_et = np.mean(np.array(gazes_perframe[i])[:, 1:], axis=0)

            if use_deepgaze:
                """
                ONLY use only "high confidence" Deepgaze points for frames in
                which a single local maximum was identified
                (>.8 max, >20 pixels of distance).
                """
                if len(DGvals_perframe[i]) == 1:
                    x_dg, y_dg = DGvals_perframe[i][0][1:]

                    dist_x.append(x_et - x_dg)
                    dist_y.append(y_et - y_dg)

                    et_x.append(x_et)
                    et_y.append(y_et)

                    dg_x.append(x_dg)
                    dg_y.append(y_dg)

                    frame_nums.append(i)

            else:
                et_x.append(x_et)
                et_y.append(y_et)

                dist_x.append(x_et - x_mass)
                dist_y.append(y_et - y_mass)

                frame_nums.append(i)

    return (et_x, et_y), (dist_x, dist_y), (dg_x, dg_y), frame_nums


def median_clean(frame_times, et_x, et_y, dist_x, dist_y):
    '''
    Within bins of 1/100 the number of frames,
    select only frames where distance between gaze and deepgaze falls within 0.6 stdev of the median
    These frames most likely reflect when deepgaze and gaze "look" at the same thing
    '''
    jump = int(len(dist_x)/100)
    idx = 0
    gap = 0.6 # interval of distances included around median, in stdev

    filtered_times = []
    filtered_x = []
    filtered_y = []
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
                        filtered_x.append(et_x[i])
                        filtered_y.append(et_y[i])

    return filtered_times, filtered_x, filtered_y, filtered_distx, filtered_disty


def make_Friends_QC_figure(
    sub: str,
    ses: str,
    run_num: str,
    deg_x: float,
    deg_y: float,
    frame_times: list,
    clean_times: list,
    dist_x: list,
    dist_y: list,
    all_times: list,
    all_x: list,
    all_x_aligned: np.array,
    all_y: list,
    all_y_aligned: np.array,
    all_conf: list,
    out_fig_path: str,
) -> None:
    """
    Export figures to assess drift correction per run
    """
    mosaic = """
        AB
        CD
        EF
        GH
        IJ
        KL
    """
    fs = (15, 24.0)

    fig = plt.figure(constrained_layout=True, figsize=fs)
    ax_dict = fig.subplot_mosaic(mosaic)
    run_dur = 770

    ax_dict["A"].scatter(all_times, all_x, s=10, color='xkcd:light grey', alpha=all_conf)
    ax_dict["A"].scatter(all_times, all_x_aligned, c=all_conf, s=10, cmap='terrain_r', alpha=0.2)
    ax_dict["A"].plot([0, run_dur], [1, 1], color='xkcd:black', linestyle='dashed', linewidth=1)
    ax_dict["A"].plot([0, run_dur], [0, 0], color='xkcd:black', linestyle='dashed', linewidth=1)
    ax_dict["A"].set_ylim(-2, 2)
    ax_dict["A"].set_xlim(0, run_dur)
    ax_dict["A"].set_title(f'{sub} task-friends {ses} {run_num} gaze_x')

    ax_dict["B"].scatter(all_times, all_y, color='xkcd:light grey', alpha=all_conf)
    ax_dict["B"].scatter(all_times, all_y_aligned, c=all_conf, s=10, cmap='terrain_r', alpha=0.2)#'xkcd:orange', alpha=all_conf)
    ax_dict["B"].plot([0, run_dur], [1, 1], color='xkcd:black', linestyle='dashed', linewidth=1)
    ax_dict["B"].plot([0, run_dur], [0, 0], color='xkcd:black', linestyle='dashed', linewidth=1)
    ax_dict["B"].set_ylim(-2, 2)
    ax_dict["B"].set_xlim(0, run_dur)
    ax_dict["B"].set_title(f'{sub} task-friends {ses} {run_num} gaze_y')

    for deg, idx in zip([1, 2, 3, 4], ["CD", "EF", "GH", "IJ"]):
        x_col = "xkcd:red" if deg == deg_x else "xkcd:black"
        p_x = apply_poly(frame_times, dist_x, deg, np.array(clean_times), anchors=[150, 150])
        ax_dict[idx[0]].scatter(frame_times, dist_x, s=10, alpha=0.4, color="xkcd:light blue")
        ax_dict[idx[0]].plot(np.array(clean_times), p_x, color=x_col, linewidth=3)
        ax_dict[idx[0]].set_ylim(-2, 2)
        ax_dict[idx[0]].set_xlim(0, run_dur)
        ax_dict[idx[0]].set_title(f'{sub} {ses} {run_num} dist_x {deg}-deg')

        y_col = "xkcd:red" if deg == deg_y else "xkcd:black"
        p_y = apply_poly(frame_times, dist_y, deg, np.array(clean_times), anchors=[150, 150])
        ax_dict[idx[1]].scatter(frame_times, dist_y, s=10, alpha=0.4, color="xkcd:light blue")
        ax_dict[idx[1]].plot(np.array(clean_times), p_y, color=y_col, linewidth=3)
        ax_dict[idx[1]].set_ylim(-2, 2)
        ax_dict[idx[1]].set_xlim(0, run_dur)
        ax_dict[idx[1]].set_title(f'{sub} {ses} {run_num} dist_y {deg}-deg')

    f_x = apply_gaussian(frame_times, dist_x, np.array(clean_times), rollwin_dur=15.0, fps=20.0, anchors=[20, 20])
    ax_dict["K"].scatter(frame_times, dist_x, s=10, alpha=0.4, color="xkcd:light blue")
    ax_dict["K"].plot(np.array(clean_times), f_x, color="xkcd:green", linewidth=3)
    ax_dict["K"].set_ylim(-2, 2)
    ax_dict["K"].set_xlim(0, run_dur)
    ax_dict["K"].set_title(f'{sub} {ses} {run_num} dist_x gaussian')

    f_y = apply_gaussian(frame_times, dist_y, np.array(clean_times), rollwin_dur=15.0, fps=20.0, anchors=[20, 20])
    ax_dict["L"].scatter(frame_times, dist_y, s=10, alpha=0.4, color="xkcd:light blue")
    ax_dict["L"].plot(np.array(clean_times), f_y, color="xkcd:green", linewidth=3)
    ax_dict["L"].set_ylim(-2, 2)
    ax_dict["L"].set_xlim(0, run_dur)
    ax_dict["L"].set_title(f'{sub} {ses} {run_num} dist_y gaussian')

    fig.savefig(out_fig_path)
    plt.close()


def driftCorr_ETfriends(
    row,
    out_path,
    use_poly=False,
    is_final=False,
) -> None:
    sub = row["subject"]
    ses = row["session"]
    run_num = row["run"]
    fnum = row["file_number"]
    task_type = row["task"]
    print(sub, ses, fnum, task_type, run_num)

    if is_final:
        bids_out_path = f'{out_path}/final_bids_DriftCor/{sub}/{ses}'
        Path(bids_out_path).mkdir(parents=True, exist_ok=True)
        out_file = f'{bids_out_path}/{sub}_{ses}_{task_type}_{run_num}_eyetrack.tsv.gz'
    else:
        outpath_fig = os.path.join(out_path, 'DC_gaze')
        Path(outpath_fig).mkdir(parents=True, exist_ok=True)
        out_file = f'{out_path}/DC_gaze/{sub}_{ses}_{run_num}_{fnum}_{task_type}_DCplot.png'

    if not os.path.exists(out_file):
        #if True:
        try:
            run_gaze = np.load(row['gaze_path'], allow_pickle=True)['gaze2d']
            all_DGvals = np.load(
                row['dg_path'],
                allow_pickle=True,
            )['deepgaze_vals']

            '''
            Get frame count, frame-per-second rate, and estimated movie
            onset time
            '''
            frame_count, zero_idx, time_zero, fps = get_indices(
                row['mkv_path'],
                run_gaze,
            )
            assert(len(all_DGvals) == frame_count)

            '''
            Realign gaze timestamps with movie onset, and filter out
            below-threshold gaze ("clean")
            '''
            gaze_threshold = row['pupilConf_thresh'] if not pd.isna(
                row['pupilConf_thresh']
            ) else 0.9
            reset_gaze_list, all_vals, clean_vals  = reset_gaze_time(
                run_gaze,
                time_zero,
                gaze_threshold,
                distance=False,
            )

            # normalized position (x and y), time (s) from onset
            # and confidence for all gaze
            all_x, all_y, all_times, all_conf = all_vals
            # normalized position (x and y) for all gaze above
            # confidence threshold
            clean_x, clean_y, clean_times, clean_conf = clean_vals

            """
            assign above-threshold uncorrected gaze to movie frames
            """
            uncorr_gazes = gaze_2_frame(
                frame_count,
                fps,
                clean_x,
                clean_y,
                clean_times,
            )

            """
            Calculate distance between frame's mean gaze and
            deepgaze's estimated coordinates
            """
            (et_x, et_y), (dist_x, dist_y), (dg_x, dg_y), frame_nums = get_distances(
                frame_count,
                uncorr_gazes,
                all_DGvals,
                use_deepgaze=True,
            )
            # convert frame numbers to seconds
            frame_times = (np.array(frame_nums) + 0.5) / fps

            # remove distances too far from median within sliding window for cleaner signal
            frame_times, et_x, et_y, dist_x, dist_y = median_clean(
                frame_times,
                et_x,
                et_y,
                dist_x,
                dist_y,
            )

            """
            Apply drift correction.
            """
            deg_x = int(row['polyDeg_x']) if not pd.isna(row['polyDeg_x']) else 4
            deg_y = int(row['polyDeg_y']) if not pd.isna(row['polyDeg_y']) else 4

            if use_poly:
                """
                Implement polynomial correction by drawing a polynomial through
                the distance between the measured gaze and the gaze estimated
                by DeepGaze.

                Default degree is 4 in x and in y. Each can be adjusted for
                each run (range [1-4]) by specifying them in the 'polyDeg_x'
                and 'polyDeg_y' columns.
                """
                anchors = [150, 150]
                # correct all x gaze
                p_of_all_x = apply_poly(
                    frame_times,
                    dist_x,
                    deg_x,
                    np.array(all_times),
                    anchors=anchors,
                )
                all_x_aligned = np.array(all_x) - (p_of_all_x)

                # correct all y gaze
                p_of_all_y = apply_poly(
                    frame_times,
                    dist_y, deg_y,
                    np.array(all_times),
                    anchors=anchors,
                )
                all_y_aligned = np.array(all_y) - (p_of_all_y)

            else:
                """
                Apply gaussian filter correction to distances between the measured
                gaze and the gaze estimated by DeepGaze: better fit than polynomial.
                """
                # correct all x gaze
                f_of_all_x = apply_gaussian(
                    frame_times,
                    dist_x,
                    np.array(all_times),
                    rollwin_dur=15.0, # 15 seconds windows
                    fps=20.0,
                    anchors=[20, 20],
                )
                all_x_aligned = np.array(all_x) - (f_of_all_x)

                # correct all y gaze
                f_of_all_y = apply_gaussian(
                    frame_times,
                    dist_y,
                    np.array(all_times),
                    rollwin_dur=15.0, # 15 seconds windows
                    fps=20.0,
                    anchors=[20, 20],
                )
                all_y_aligned = np.array(all_y) - (f_of_all_y)

            if is_final:
                df_gaze = format_gaze(
                    all_x_aligned,
                    all_y_aligned,
                    all_times,
                    reset_gaze_list,
                )

                df_gaze.to_csv(
                    out_file,
                    sep='\t',
                    header=True,
                    index=False,
                    compression='gzip',
                )

            else:
                '''
                plot QC figures (one per run) to assess drift correction and
                select correction strategy
                '''
                make_Friends_QC_figure(
                    sub,
                    ses,
                    run_num,
                    deg_x,
                    deg_y,
                    frame_times,
                    clean_times,
                    dist_x,
                    dist_y,
                    all_times,
                    all_x,
                    all_x_aligned,
                    all_y,
                    all_y_aligned,
                    all_conf,
                    out_file,
                )

        except:
            print('could not process')


def main():
    '''
    This script performs drift-correction on the Friends eyetracking data using
    framewise gaze predictions from DeepgazeMr.
    '''
    args = get_arguments()
    # e.g., (elm): /unf/eyetracker/neuromod/triplets/sourcedata
    in_path = args.in_path
    out_path = args.out_path
    mkv_path = args.mkv_path
    #mkv_path = "/data/neuromod/DATA/cneuromod/friends/stimuli/s1/friends_s01e13a.mkv"
    use_poly = args.use_poly
    is_final = args.is_final

    # load list of valid files (those deserialized and exported as npz in step 2 that passed QC)
    clean_list = get_list(out_path, is_final)
    clean_list = add_file_paths(clean_list, in_path, out_path, mkv_path, 'friends')

    # implement drift correction on each run
    clean_list.apply(lambda row: driftCorr_ETfriends(
        row,
        out_path,
        use_poly,
        is_final,
    ), axis=1)


if __name__ == '__main__':
    sys.exit(main())
