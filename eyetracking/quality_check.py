import os, sys, platform, json
import numpy as np
from types import SimpleNamespace

import math
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Perform off-line gaze mapping with 2D and 3D pupil detectors ')
parser.add_argument('--run_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--config', default='config.json', type=str, help='absolute path to config file')
args = parser.parse_args()

sys.path.append(os.path.join(args.run_dir, "pupil", "pupil_src", "shared_modules"))
#sys.path.append(os.path.join("/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking", "pupil", "pupil_src", "shared_modules"))
from video_capture.file_backend import File_Source
from file_methods import PLData_Writer, load_pldata_file, load_object, save_object
from gaze_producer.worker.fake_gpool import FakeGPool, FakeIPC

from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
from gaze_mapping.gazer_2d import Gazer2D
from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
from gaze_mapping.gazer_3d.gazer_headset import Gazer3D

#from gaze_mapping.utils import _find_nearest_idx as find_idx

'''
Quality checks

1. Flag missing frames based on gaps in timestamps
2. Assess difference in position between online 2d pupil, offline 2d pupil and offline 3d pupil over time / course of run
3. Assess difference in gaze position between online 2d pupil, offline 2d pupil and offline 3d pupil over time / course of run
4. If calibration data, assess distance between gaze position and markers position
5. Assess model drift over time: distance between gaze (median within sliding window) and center of the screen. Or just plot median position over time to estimate drift
'''


def make_plot(rs, rois, out_dir=None, out_name=''):

    ncols = 4
    nrows = math.ceil(len(rois) / ncols)

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, nrows * 4))

    ct = 0
    for r in range(nrows):
        for c in range(ncols):
            if ct == len(rois):
                break

            roi = rois[ct]
            ct += 1

            roi_rs = rs[roi][0]

            if np.isnan(roi_rs).any():
                roi_rs = np.zeros(roi_rs.shape)

            n_bins = 30
            N, bins, patches = ax[r][c].hist(roi_rs, edgecolor='white', linewidth=1, bins=n_bins)

            insign_perc = min([(0.18 - np.array(roi_rs).min()) / (np.array(roi_rs).max() - np.array(roi_rs).min()), 1])
            try:
                insign_bins = int(n_bins * insign_perc)
            except:
                insign_bins = n_bins

            for i in range(0, insign_bins):
                patches[i].set_facecolor('lightgray')
            for i in range(insign_bins, 30):
                patches[i].set_facecolor('lightblue')

            ax[r][c].axvline(x=np.median([x for x in roi_rs if x >= 0.18]), color='red', linestyle='--')

            ax[r][c].set_title('{} (n={}, max={:.2f}, median={:.2f})'.format(roi, len(roi_rs), np.array(roi_rs).max(),
                                                                             np.median(
                                                                                 [x for x in roi_rs if x >= 0.18])), pad=30)

            ax[r][c].spines['top'].set_visible(False)
            ax[r][c].spines['right'].set_visible(False)

    fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Pearson correlation", labelpad=20)
    plt.ylabel("Number of voxels", labelpad=20)

    fig.tight_layout(pad=6.0)

    if out_dir is not None:
        #plt.savefig('{}/correlation_histograms.png'.format(out_dir))
        plt.savefig(out_dir + '/' + out_name + 'correlation_histograms.png')



def match_batch(pupils1, pupils2, max_dispersion=1 / 15.0):
    """Get pupil positions closest in time to ref points.
    Return list of dict with matching ref and pupil datum.
    """
    '''
    matched = [[], []]

    pupils1_ts = np.array([p["timestamp"] for p in pupils1])

    for pup in pupils2:
        closest_p_idx = find_idx(pupils1_ts, pup["timestamp"])
        closest_p = pupils1[closest_p_idx]
        dispersion = max(closest_p["timestamp"], pup["timestamp"]) - min(
            closest_p["timestamp"], pup["timestamp"]
        )
        if dispersion < max_dispersion:
            matched[0].append(pup)
            matched[1].append(closest_p)

    return matched
    '''

    matched_pup1 = []
    matched_pup2 = []

    i = 0

    for pup in pupils1:
        unfound = True
        while unfound:
            if i == len(pupils2):
                unfound = False
            elif abs(pup['timestamp'] - pupils2[i]['timestamp']) < 0.001:
                matched_pup1.append(pup)
                matched_pup2.append(pupils2[i])
                i += 1
                unfound = False
            else:
                i += 1

    return matched_pup1, matched_pup2


def assess_timegaps(t_stamps, threshold = 0.004016):
    '''
    Input:
        t_stamps (list of times): list of pupil time stamps per frames
        thresh (float): time gap (in ms) between frames above which a "freeze" is reported,
    Output:
        time_diff: list of floats that reflect the time difference between a frame and the subsequent one
        skip_idx (list of int): list of frame indices where an above-threshold time gape is recorded
    '''
    time_diff = []
    skip_idx = []

    for i in range(1, len(t_stamps)):
        diff = t_stamps[i] - t_stamps[i-1]
        time_diff.append(diff)

        if diff > threshold:
            skip_idx.append(i-1, diff, t_stamps[i-1])

    return time_diff, skip_idx


def assess_pupil_freezes(pupils, thresh = 0.004016):
    '''
    Input:
        pupils (list of dict): list of pupil data per frames
        thresh (float): time gap (in ms) between frames above which a "freeze" is reported,
    Output:
        tuple (time_diff, skip_idx): time differences and frame indices where above threshold time gaps
    '''
    t_stamps = []

    for i in range(len(c_pupils)):
        t_stamps.append(c_pupils[i]['timestamp'])

    return assess_timegaps(t_stamps, thresh)


def assess_distance(pupils1, pupils2):
    '''
    Distance in pixels, assumes square pixels
    '''
    dist_list = []

    print(len(pupils1), len(pupils2))
    # match time stamps between files;
    # offline pupils start and stop later, for some odd reason...
    if pupils1[0]['timestamp'] < pupils2[0]['timestamp']:
        pupils2, pupils1 = match_batch(pupils2, pupils1)
    else:
        pupils1, pupils2 = match_batch(pupils1, pupils2)

    assert len(pupils1) == len(pupils2)
    print(len(pupils1), len(pupils2))

    for i in range(len(pupils1)):
        assert abs(pupils1[i]['timestamp'] - pupils2[i]['timestamp']) < 0.001
        x_1, y_1 = pupils1[i]['ellipse']['center'] # norm_pos (in % screen), center (in pixels)
        x_2, y_2 = pupils2[i]['ellipse']['center']

        dist = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
        dist_list.append(dist)

    return dist_list


if __name__ == "__main__":

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    g_pool = SimpleNamespace()

    eye_file = File_Source(g_pool, cfg['mp4'])
    t_stamps = eye_file.timestamps

    diff_list, gap_idx = assess_timegaps(t_stamps, cfg['time_threshold'])

    if len(gap_idx) > 0:
        #export as .tsv
        np.savetxt(os.path.join(cfg['out_dir'], 'frame_gaps.tsv'), np.array(gap_idx), delimiter="\t")


    online_pupils = load_pldata_file(cfg['online_pupils'], 'pupil')[0]
    offline_pupils2d = np.load(cfg['offline_pupils2D'], allow_pickle=True)['pupils2d']
    offline_pupils3d = np.load(cfg['offline_pupils3D'], allow_pickle=True)['pupils3d']

    diff_on2d_off2d = assess_distance(online_pupils, offline_pupils2d)
    diff_off2d_off3d = assess_distance(offline_pupils2d, offline_pupils3d)
    diff_on2d_off3d = assess_distance(online_pupils, offline_pupils3d)

    # Export lists of difference as dictionary (.npz file w 3 entries)
    np.savez(os.path.join(cfg['out_dir'], 'pupil_differences.npz'), diff_on2d_off2d = np.array(diff_on2d_off2d),diff_off2d_off3d = np.array(diff_off2d_off3d),diff_on2d_off3d = np.array(diff_on2d_off3d))

    # Make and export plots
