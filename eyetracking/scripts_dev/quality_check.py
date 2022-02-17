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
Quality checks: contrast two sets of pupils and gaze

1. (Optional) Flag missing frames in eye movie (mp4) based on gaps in camera timestamps
2. Assess difference in position between two sets of pupils over time / course of run
3. Assess difference in position (mapping) between two sets of gaze over time / course of run
'''


def export_line_plot(y_val, out_name=None, x_val=None):
    plt.clf()
    if x_val is not None:
        plt.plot(x_val, y_val)
    else:
        plt.plot(y_val)

    if out_name is not None:
        #plt.savefig('{}/correlation_histograms.png'.format(out_dir))
        plt.savefig(out_name)


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

'''


def match_batch(dset1, dset2, max_dispersion=1 / 15.0):
    """Get pupil or gaze positions closest in time to ref points.
    Return list of dict with matching ref and pupil / gaze datum.
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

    matched_set1 = []
    matched_set2 = []

    i = 0

    for dpoint in dset1:
        unfound = True
        while unfound:
            if i == len(dset2):
                unfound = False
            elif abs(dpoint['timestamp'] - dset2[i]['timestamp']) < 0.001:
                matched_set1.append(dpoint)
                matched_set2.append(dset2[i])
                i += 1
                unfound = False
            else:
                i += 1

    return matched_set1, matched_set2


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
            skip_idx.append([i-1, diff, t_stamps[i-1]])

    return time_diff, skip_idx


def qc_report(list_data, output_name, data_type, cf_thresh=0.6, tg_thresh = 0.004016):
    '''
    Input:
        list_data (list of dict): list of pupil or gaze data per frames
        output_name (string): path and name of output file
        cf_thresh (float): confidence threshold
        tg_thresh (float): time gap (in ms) between frames above which a "freeze" is reported,
    Output:
        tuple (time_diff, skip_idx): time differences and frame indices where above threshold time gaps
    '''

    t_stamps = []
    confidences = []
    positions = []

    for i in range(len(list_data)):
        tstamp = list_data[i]['timestamp']
        t_stamps.append(tstamp)
        cf = list_data[i]['confidence']
        confidences.append(cf)
        if cf > cf_thresh:
            if data_type == 'pupils':
                x, y = list_data[i]['ellipse']['center']
            elif data_type == 'gaze':
                x, y = list_data[i]['norm_pos']
                x = 1.0 if x > 1.0 else x
                x = 0.0 if x < 0.0 else x
                y = 1.0 if y > 1.0 else y
                y = 0.0 if y < 0.0 else y
            positions.append([x, y, tstamp])

    print(os.path.basename(output_name) + ' has ' + str(100*(1 - len(positions)/len(list_data))) + '% of frames below confidence threshold')

    time_diff, skip_idx = assess_timegaps(t_stamps, tg_thresh)

    x = np.array(positions)[:, 0].tolist()
    y = np.array(positions)[:, 1].tolist()
    times = np.array(positions)[:, 2].tolist()

    export_line_plot(confidences, output_name + '_confidence.png')
    export_line_plot(x, output_name + '_Xposition.png', times)
    export_line_plot(y, output_name + '_Yposition.png', times)
    #export_line_plot(time_diff, output_name + '_timediff.png')

    if data_type == 'gaze':
        w_size = 100
        extra = len(positions) % w_size
        x_medians = np.median(np.reshape(x[:-extra], (-1, w_size)), axis=0)
        y_medians = np.median(np.reshape(y[:-extra], (-1, w_size)), axis=0)
        export_line_plot(x_medians, output_name + '_Xmedians.png')
        export_line_plot(y_medians, output_name + '_Ymedians.png')
    '''
    if len(skip_idx) > 0:
        print(os.path.basename(output_name) + ' has ' + str(len(skip_idx)) + ' time gaps')
        np.savez(output_name + '_QCrep.npz', confidence = np.array(confidences), position = np.array(positions), time_diff = np.array(time_diff), time_gaps = np.array(skip_idx))
    else:
        np.savez(output_name + '_QCrep.npz', confidence = np.array(confidences), position = np.array(positions), time_diff = np.array(time_diff))
    '''

def assess_distance(out_name, dtype, dset1, dset2, cf_thresh=0.6):
    '''
    Distance in pixels between center of pupil's position on eye movie frames;
    Assumes square pixels

    Or distance in normalized space between gaze positions
    '''
    out_name = out_name + '_' + dtype

    dist_list = []
    time_list = []

    print(len(dset1), len(dset2))
    # match time stamps between files;
    # offline pupils start and stop later, for some odd reason...
    if dset1[0]['timestamp'] < dset2[0]['timestamp']:
        dset2, dset1 = match_batch(dset2, dset1)
    else:
        dset1, dset2 = match_batch(dset1, dset2)

    assert len(dset1) == len(dset2)
    print(len(dset1), len(dset2))

    for i in range(len(dset1)):
        assert abs(dset1[i]['timestamp'] - dset2[i]['timestamp']) < 0.001
        tstamp = dset1[i]['timestamp']
        c1 = dset1[i]['confidence']
        c2 = dset2[i]['confidence']
        if (c1 > cf_thresh) and (c2 > cf_thresh):
            if dtype == 'pupils':
                x_1, y_1 = dset1[i]['ellipse']['center'] # norm_pos (in % screen), center (in pixels)
                x_2, y_2 = dset2[i]['ellipse']['center']
            elif dtype == 'gaze':
                x_1, y_1 = dset1[i]['norm_pos'] # norm_pos (in % screen), center (in pixels)
                x_2, y_2 = dset2[i]['norm_pos']
            dist = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
            dist_list.append(dist)
            time_list.append(tstamp)

    # Export lists of difference as dictionary (.npz file w 3 entries)
    #np.savez(os.path.join(cfg['out_dir'], cfg['out_name'] + 'pupil_differences.npz'), diff_on2d_off2d = np.array(diff_on2d_off2d),diff_off2d_off3d = np.array(diff_off2d_off3d),diff_on2d_off3d = np.array(diff_on2d_off3d))

    # Export lists of difference as dictionary (.npz file w single entry)
    '''
    np.savez(out_name + '_distance.npz', distance = np.array(dist_list), t_stamps = np.array(time_list))
    '''
    export_line_plot(dist_list, out_name + '_distance.png', time_list)


if __name__ == "__main__":
    '''
    Performs Quality Check and comparisons between two different sets of pupils and their corresponding gaze
    for a certain analysis (run)
    '''

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    '''
    Step 1. Assess eye camera freezing: any time gaps in in-scan eye capture? (from in-scan eye mp4)
    '''
    if 'mp4' in cfg:
        g_pool = SimpleNamespace()
        eye_file = File_Source(g_pool, cfg['mp4'])
        t_stamps = eye_file.timestamps

        diff_list, gap_idx = assess_timegaps(t_stamps, cfg['time_threshold'])

        if len(gap_idx) > 0:
            #export as .tsv
            np.savetxt(os.path.join(cfg['out_dir'], cfg['out_name'] + '_frame_gaps.tsv'), np.array(gap_idx), delimiter="\t")

    '''
    Step 2. QC on pupils
    For each set (separately)
        Assess number of missing pupils (based on intervals in time stamps)
        Plot confidence and position in x and y over time
    Contrast the two sets to one another: difference in pupil positions (in pixels)
    '''
    # Load first set of pupils
    # Note that online pupils are always captured w a 2D model in our set up
    if 'pupils1' in cfg:
        if cfg['isOnline_pupils1']:
            pupils1 = load_pldata_file(cfg['pupils1'], 'pupil')[0]
        else:
            p1_tag = 'pupils3d' if cfg['is3D_pupils1'] else 'pupils2d'
            pupils1 = np.load(cfg['pupils1'], allow_pickle=True)[p1_tag]

        qc_report(pupils1, os.path.join(cfg['out_dir'], cfg['pupils1_name']), 'pupils', cfg['pupil_confidence_threshold'])

    # Load second set of pupils
    if 'pupils2' in cfg:
        if cfg['isOnline_pupils2']:
            pupils2 = load_pldata_file(cfg['pupils2'], 'pupil')[0]
        else:
            p2_tag = 'pupils3d' if cfg['is3D_pupils2'] else 'pupils2d'
            pupils2 = np.load(cfg['pupils2'], allow_pickle=True)[p2_tag]

        qc_report(pupils2, os.path.join(cfg['out_dir'], cfg['pupils2_name']), 'pupils', cfg['pupil_confidence_threshold'])

    # Contrast two sets of pupils to one another
    if 'pupils1' in cfg and 'pupils2' in cfg:
        assess_distance(os.path.join(cfg['out_dir'], cfg['out_name']), 'pupils', pupils1, pupils2, cfg['pupil_confidence_threshold'])

    '''
    Step 3. Contrast gaze positions
    '''
    if 'gaze1' in cfg:
        if cfg['isOnline_gaze1']:
            gaze1 = load_pldata_file(cfg['gaze1'], 'gaze')[0]
        else:
            g1_tag = 'gaze3d' if cfg['is3D_gaze1'] else 'gaze2d'
            gaze1 = np.load(cfg['gaze1'], allow_pickle=True)[g1_tag]

        qc_report(gaze1, os.path.join(cfg['out_dir'], cfg['gaze1_name']), 'gaze', cfg['gaze_confidence_threshold'])

    if 'gaze2' in cfg:
        if cfg['isOnline_gaze2']:
            gaze2 = load_pldata_file(cfg['gaze2'], 'gaze')[0]
        else:
            g2_tag = 'gaze3d' if cfg['is3D_gaze2'] else 'gaze2d'
            gaze2 = np.load(cfg['gaze2'], allow_pickle=True)[g2_tag]

        qc_report(gaze2, os.path.join(cfg['out_dir'], cfg['gaze2_name']), 'gaze', cfg['gaze_confidence_threshold'])

    # Contrast two sets of gaze to one another
    if 'gaze1' in cfg and 'gaze2' in cfg:
        assess_distance(os.path.join(cfg['out_dir'], cfg['out_name']), 'gaze', gaze1, gaze2, cfg['gaze_confidence_threshold'])
