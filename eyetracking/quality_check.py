import os, sys, platform, json
import numpy as np
from types import SimpleNamespace

sys.path.append(os.path.join("/home/labopb/Documents/Marie/neuromod/pupil", "pupil_src", "shared_modules"))
from video_capture.file_backend import File_Source
from file_methods import PLData_Writer, load_pldata_file, load_object, save_object
from gaze_producer.worker.fake_gpool import FakeGPool, FakeIPC

from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
from gaze_mapping.gazer_2d import Gazer2D
from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
from gaze_mapping.gazer_3d.gazer_headset import Gazer3D

import argparse

parser = argparse.ArgumentParser(description='Perform off-line gaze mapping with 2D and 3D pupil detectors ')
parser.add_argument('--data_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--config', default='config.json', type=str, help='absolute path to config file')
args = parser.parse_args()


'''
Quality checks

1. Flag missing frames based on gaps in timestamps
2. Assess difference in position between online 2d pupil, offline 2d pupil and offline 3d pupil over time / course of run
3. Assess distance between gaze position and markers position
4. Assess model drift over time: distance between gaze (median within sliding window) and center of the screen. Or just plot median position over time to estimate drift
'''

'''Load pupil file'''


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

        if diff > thresh:
            skip_idx.append(i-1)

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

    dist_list = []

    for i in range(len(pupils1)):
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
        #export as npy
        pass

    # export diff_list?

    online_pupils = load_pldata_file(cfg['online_pupils'], 'pupil')
    # TODO: convert serialized data to list
    offline_pupils2d = np.load(cfg['offline_pupils2D'], allow_pickle=True)
    offline_pupils3d = np.load(cfg['offline_pupils3D'], allow_pickle=True)

    assert len(online_pupils) == len(offline_pupils2d)
    assert len(offline_pupils2d) == len(offline_pupils3d)

    diff_on_off2d = assess_distance(online_pupils, offline_pupils2d)
    diff_off2d_off3d = assess_distance(offline_pupils2d, offline_pupils3d)
    diff_on_off3d = assess_distance(online_pupils, offline_pupils3d)

    # Export lists of difference as dictionary (.npz file w 3 entries)

    # Make and export plots
