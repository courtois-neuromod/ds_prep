import os, sys, platform, json
import numpy as np
import pandas as pd
from types import SimpleNamespace

import math
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Perform off-line gaze mapping with 2D and 3D pupil detectors ')
parser.add_argument('--code_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--config', default='config.json', type=str, help='absolute path to config file')
args = parser.parse_args()

sys.path.append(os.path.join(args.code_dir, "pupil", "pupil_src", "shared_modules"))
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
Quality checks for runs of THINGS data

An early version, exports too many charts. Use THINGS_qualitycheck_summary for more concise output

1. (Optional) Flag missing frames in eye movie (mp4) based on gaps in camera timestamps
2. Flag percentage of pupils and gaze under confidence threshold
3. Flag percentage of gaze outside screen area
4. Plot x and z pupil and gaze position over time
'''

def make_detection_gpool():
    g_pool = SimpleNamespace()

    rbounds = SimpleNamespace()
    # TODO: optimize? Narrow down search window?
    rbounds.bounds = (0, 0, 640, 480) # (minx, miny, maxx, maxy)
    g_pool.roi = rbounds

    g_pool.display_mode = "algorithm" # "roi" # for display; doesn't change much
    g_pool.eye_id = 0 #'eye0'
    g_pool.app = "player" # "capture"

    return g_pool


def export_line_plot(y_val, out_name=None, x_val=None, mid_val=0.0, x_range=None, y_range=None):

    plt.clf()
    if x_val is not None:
        x_val = np.array(x_val)
        x_0 = x_val[0]
        x_val -= x_0
        m, b = np.polyfit(x_val, y_val, 1)
        p3, p2, p1, p0 = np.polyfit(x_val, y_val, 3)
        #p6, p5, p4, p3, p2, p1, p0 = np.polyfit(x_val, y_val, 6)
        plt.plot(x_val, y_val)
        #plt.plot(x_val, m*x_val + b)
        p_of_x = p3*(x_val**3) + p2*(x_val**2) + p1*(x_val) + p0
        #p_of_x = p6*(x_val**6) + p5*(x_val**5) + p4*(x_val**4) + p3*(x_val**3) + p2*(x_val**2) + p1*(x_val) + p0
        diff_to_mid = np.average((p_of_x - mid_val)**2)
        plt.plot(x_val, p_of_x)
    else:
        diff_to_mid = m = b = -1
        plt.plot(y_val)

    if x_range is not None:
        plt.xlim(x_range)

    if y_range is not None:
        plt.ylim(y_range)

    if out_name is not None:
        #plt.savefig('{}/correlation_histograms.png'.format(out_dir))
        plt.savefig(out_name)

    return diff_to_mid, m, b

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
            print('skipped frames')
            print([i-1, diff, t_stamps[i-1]])
            skip_idx.append([i-1, diff, t_stamps[i-1]])

    return time_diff, skip_idx


def qc_report(list_data, output_path, output_name, data_type, cf_thresh=0.6, tg_thresh = 0.004016):
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

    off_bounds = 0

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

                is_off = 0
                if x > 1.0:
                    x = 1.0
                    is_off = 1
                elif x < 0.0:
                    x = 0.0
                    is_off = 1
                if y > 1.0:
                    y = 1.0
                    is_off = 1
                elif y < 0.0:
                    y = 0.0
                    is_off = 1
                off_bounds += is_off
                #x = 1.0 if x > 1.0 else x
                #x = 0.0 if x < 0.0 else x
                #y = 1.0 if y > 1.0 else y
                #y = 0.0 if y < 0.0 else y
            positions.append([x, y, tstamp])

    #s1 = os.path.basename(output_name) + ' has ' + str(100*(1 - len(positions)/len(list_data))) + '% of ' + data_type + ' below' + str(cf_thresh) + 'confidence threshold'
    d1 = [output_name, 100*(1 - len(positions)/len(list_data))]
    s1 = output_name + ' has ' + str(d1[1]) + '% of ' + data_type + ' below ' + str(cf_thresh) + ' confidence threshold'
    if data_type == 'gaze':
        #s1 += '\n' + os.path.basename(output_name) + ' has ' + str(100*(off_bounds/len(list_data))) + '% of gaze positions outside screen area'
        d1.append(100*(off_bounds/len(list_data)))
        s1 += '\n' + output_name + ' has ' + str(d1[2]) + '% of gaze positions outside screen area'
    print(s1)

    time_diff, skip_idx = assess_timegaps(t_stamps, tg_thresh)

    x = np.array(positions)[:, 0].tolist()
    y = np.array(positions)[:, 1].tolist()
    times = np.array(positions)[:, 2].tolist()
    _, _, _ = export_line_plot(confidences, os.path.join(output_path, 'confidence_' + output_name + '.png'), y_range=[-0.05, 1.05])
    #export_line_plot(time_diff, output_name + '_timediff.png')

    if data_type == 'gaze':
        xdiff, x_m, x_b = export_line_plot(x, os.path.join(output_path, 'Xposition_' + output_name + '.png'), times, mid_val=0.5, y_range=[-0.05, 1.05])
        ydiff, y_m, y_b = export_line_plot(y, os.path.join(output_path, 'Yposition_' + output_name + '.png'), times, mid_val=0.5, y_range=[-0.05, 1.05])

        w_size = 100
        extra = len(positions) % w_size
        #x_medians = np.median(np.reshape(x[:-extra], (-1, w_size)), axis=0)
        #y_medians = np.median(np.reshape(y[:-extra], (-1, w_size)), axis=0)
        #export_line_plot(x_medians, os.path.join(output_path, 'Xmedians_' + output_name + '.png'), y_range=[-0.05, 1.05])
        #export_line_plot(y_medians, os.path.join(output_path, 'Ymedians_'+ output_name + '.png'), y_range=[-0.05, 1.05])

    elif data_type == 'pupils':
        xdiff, x_m, x_b = export_line_plot(x, os.path.join(output_path, 'Xposition_' + output_name + '.png'), times, mid_val=320, y_range=[-4.0, 644.0])
        ydiff, y_m, y_b = export_line_plot(y, os.path.join(output_path, 'Yposition_' + output_name + '.png'), times, mid_val=240, y_range=[-4.0, 484.0])

    '''
    if len(skip_idx) > 0:
        print(os.path.basename(output_name) + ' has ' + str(len(skip_idx)) + ' time gaps')
        np.savez(output_name + '_QCrep.npz', confidence = np.array(confidences), position = np.array(positions), time_diff = np.array(time_diff), time_gaps = np.array(skip_idx))
    else:
        np.savez(output_name + '_QCrep.npz', confidence = np.array(confidences), position = np.array(positions), time_diff = np.array(time_diff))
    '''
    return s1, d1, (xdiff, ydiff, x_m, x_b, y_m, y_b)

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
    #export_line_plot(dist_list, out_name + '_distance.png', time_list)
    export_line_plot(dist_list, out_name + '_distance.png', time_list)


def old_main():

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

        p1s, p1d, _ = qc_report(pupils1, cfg['out_dir'], cfg['pupils1_name'], 'pupils', cfg['pupil_confidence_threshold'])

    # Load second set of pupils
    if 'pupils2' in cfg:
        if cfg['isOnline_pupils2']:
            pupils2 = load_pldata_file(cfg['pupils2'], 'pupil')[0]
        else:
            p2_tag = 'pupils3d' if cfg['is3D_pupils2'] else 'pupils2d'
            pupils2 = np.load(cfg['pupils2'], allow_pickle=True)[p2_tag]

        p2s, p2d, _ = qc_report(pupils2, cfg['out_dir'], cfg['pupils2_name'], 'pupils', cfg['pupil_confidence_threshold'])

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

        g1s, g1d, _ = qc_report(gaze1, cfg['out_dir'], cfg['gaze1_name'], 'gaze', cfg['gaze_confidence_threshold'])

    if 'gaze2' in cfg:
        if cfg['isOnline_gaze2']:
            gaze2 = load_pldata_file(cfg['gaze2'], 'gaze')[0]
        else:
            g2_tag = 'gaze3d' if cfg['is3D_gaze2'] else 'gaze2d'
            gaze2 = np.load(cfg['gaze2'], allow_pickle=True)[g2_tag]

        p2s, p2d, _ = qc_report(gaze2, cfg['out_dir'], cfg['gaze2_name'], 'gaze', cfg['gaze_confidence_threshold'])

    # Contrast two sets of gaze to one another
    if 'gaze1' in cfg and 'gaze2' in cfg:
        assess_distance(os.path.join(cfg['out_dir'], cfg['out_name']), 'gaze', gaze1, gaze2, cfg['gaze_confidence_threshold'])


def map_run_gaze(cfg, run):

    gct = str(cfg['gaze_confidence_threshold'])
    pct = str(cfg['pupil_confidence_threshold'])

    gaze_report = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + gct + ' Confidence Threshold', 'Outside Screen Area', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept'])
    pupil_report = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + pct + ' Confidence Threshold', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept'])

    run_report = open(os.path.join(cfg['out_dir'], 'qc', 'run' + run + '_report.txt'), 'w+')

    # check for missing frames in calibration eye movie (mp4)
    g_pool = make_detection_gpool()
    calib_eye_file = File_Source(g_pool, source_path=cfg['run' + run + '_calib_mp4'])
    calib_t_stamps = calib_eye_file.timestamps
    diff_list, gap_idx = assess_timegaps(calib_t_stamps, cfg['time_threshold'])
    if len(gap_idx) > 0:
        #export as .tsv
        np.savetxt(os.path.join(cfg['out_dir'], 'qc', 'run' + run + '_calib_framegaps.tsv'), np.array(gap_idx), delimiter="\t")

    # check for missing frames in main run eye movie (mp4)
    g_pool = make_detection_gpool()
    run_eye_file = File_Source(g_pool, source_path=cfg['run' + run + '_run_mp4'])
    run_t_stamps = run_eye_file.timestamps
    diff_list, gap_idx = assess_timegaps(run_t_stamps, cfg['time_threshold'])
    if len(gap_idx) > 0:
        #export as .tsv
        np.savetxt(os.path.join(cfg['out_dir'], 'qc', 'run' + run + '_run_framegaps.tsv'), np.array(gap_idx), delimiter="\t")

    # QC online pupils from calibration sequence
    calib_online_pupils = load_pldata_file(cfg['run' + run + '_calib_mp4'][:-9], 'pupil')[0]
    cp_on2d_s, cp_on2d_d, (xdiff, ydiff, x_m, x_b, y_m, y_b) = qc_report(calib_online_pupils, cfg['out_dir'] + '/qc', 'pupil_calib_online2D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
    run_report.write(cp_on2d_s + '\n')
    cp_on2d_d = [cp_on2d_d[0], 'Calib', 'Online2D', 'Run' + run, cp_on2d_d[1], xdiff, ydiff, x_m, x_b, y_m, y_b]
    pupil_report = pupil_report.append(pd.Series(cp_on2d_d, index=pupil_report.columns), ignore_index=True)

    # QC online gaze from calibration sequence
    calib_online_gaze = load_pldata_file(cfg['run' + run + '_calib_mp4'][:-9], 'gaze')[0]
    cg_on2d_s, cg_on2d_d, (xdiff, ydiff, x_m, x_b, y_m, y_b) = qc_report(calib_online_gaze, cfg['out_dir'] + '/qc', 'gaze_calib_online2D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
    run_report.write(cg_on2d_s + '\n')
    cg_on2d_d = [cg_on2d_d[0], 'Calib', 'Online2D', 'Run' + run, cg_on2d_d[1], cg_on2d_d[2], xdiff, ydiff, x_m, x_b, y_m, y_b]
    gaze_report = gaze_report.append(pd.Series(cg_on2d_d, index=gaze_report.columns), ignore_index=True)

    # QC online pupils from main run (task)
    run_online_pupils = load_pldata_file(cfg['run' + run + '_run_mp4'][:-9], 'pupil')[0]
    rp_on2d_s, rp_on2d_d, (xdiff, ydiff, x_m, x_b, y_m, y_b) = qc_report(run_online_pupils, cfg['out_dir'] + '/qc', 'pupil_run_online2D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
    run_report.write(rp_on2d_s + '\n')
    rp_on2d_d = [rp_on2d_d[0], 'Run', 'Online2D', 'Run' + run, rp_on2d_d[1], xdiff, ydiff, x_m, x_b, y_m, y_b]
    pupil_report = pupil_report.append(pd.Series(rp_on2d_d, index=pupil_report.columns), ignore_index=True)

    # QC online gaze from main run (task)
    run_online_gaze = load_pldata_file(cfg['run' + run + '_run_mp4'][:-9], 'gaze')[0]
    rg_on2d_s, rg_on2d_d, (xdiff, ydiff, x_m, x_b, y_m, y_b) = qc_report(run_online_gaze, cfg['out_dir'] + '/qc', 'gaze_run_online2D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
    run_report.write(rg_on2d_s + '\n')
    rg_on2d_d = [rg_on2d_d[0], 'Run', 'Online2D', 'Run' + run, rg_on2d_d[1], rg_on2d_d[2], xdiff, ydiff, x_m, x_b, y_m, y_b]
    gaze_report = gaze_report.append(pd.Series(rg_on2d_d, index=gaze_report.columns), ignore_index=True)

    run_report.close()

    return pupil_report, gaze_report



if __name__ == "__main__":
    '''
    Script performs quality check for online pupil and gaze outputs for all runs
    from a single THINGS session.

    Note that the same config file can be used to run THINGS_offline_calibration.py, which
    outputs offline pupil and gaze measures (in 2d and optionally in 3d)
    '''

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    pct = str(cfg['pupil_confidence_threshold'])
    gct = str(cfg['gaze_confidence_threshold'])

    pupil_reports = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + pct + ' Confidence Threshold', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept'])
    gaze_reports = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + gct + ' Confidence Threshold', 'Outside Screen Area', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept'])

    for run in cfg['runs']:
        print('Run ' + str(run))
        try:
            pupil_report, gaze_report = map_run_gaze(cfg, run)
            pupil_reports = pd.concat((pupil_reports, pupil_report), ignore_index=True)
            gaze_reports = pd.concat((gaze_reports, gaze_report), ignore_index=True)
        except:
            print('Something went wrong processing run ' + run)

    pupil_reports.to_csv(cfg['out_dir'] +'/qc/' + cfg['subject'] + '_ses' + cfg['session'] + '_pupil_report.tsv', sep='\t', header=True, index=False)
    gaze_reports.to_csv(cfg['out_dir'] +'/qc/' + cfg['subject'] + '_ses' + cfg['session'] + '_gaze_report.tsv', sep='\t', header=True, index=False)
