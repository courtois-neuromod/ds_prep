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
Quality check summary: exports summary statistics and plots gaze along X and Y dimensions for each run

1. (Optional) Flag missing frames in eye movie (mp4) based on gaps in camera timestamps
2. Flag percentage of pupils and gaze under confidence threshold
3. Flag percentage of gaze outside screen area
4. Calculate metrics of distance from fixation point
5. Plot x and y gaze position over time
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


def make_frameskip_figure(good_runs, outdir=None):

    calib_skips = pd.DataFrame(columns=['Run', 'Frame', 'Gap', 'Timestamp'])
    run_skips = pd.DataFrame(columns=['Run', 'Frame', 'Gap', 'Timestamp'])
    cx_runs = []
    rx_runs = []
    cx_labels = []
    rx_labels = []

    for i in range(len(good_runs)):
        try:
            calib_framegaps = pd.read_csv(os.path.join(outdir, 'run' + str(good_runs[i]) + '_calib_framegaps.tsv'), sep = '\t')
            calib_framegaps.columns = ['Run', 'Frame', 'Gap', 'Timestamp']
            calib_skips = pd.concat((calib_skips, calib_framegaps), ignore_index=True)
            cx_runs.append(int(good_runs[i]))
            cx_labels.append('Run ' + str(good_runs[i]) + ' (n=' + str(calib_framegaps.shape[0]) + ')')
        except:
            print('No calibration file for run ' + str(good_runs[i]))

        try:
            run_framegaps = pd.read_csv(os.path.join(outdir, 'run' + str(good_runs[i]) + '_run_framegaps.tsv'), sep = '\t')
            run_framegaps.columns = ['Run', 'Frame', 'Gap', 'Timestamp']
            run_skips = pd.concat((run_skips, run_framegaps), ignore_index=True)
            rx_runs.append(int(good_runs[i]))
            rx_labels.append('Run ' + str(good_runs[i]) + ' (n=' + str(run_framegaps.shape[0]) + ')')
        except:
            print('No run file for run ' + str(good_runs[i]))

    if calib_skips.shape[0] > 0:
        plt.clf()

        x_val = calib_skips['Run'].to_numpy(dtype='float16')
        y_val = calib_skips['Gap'].to_numpy(dtype='float16')
        plt.scatter(x_val, y_val, alpha=0.4)
        plt.xticks(cx_runs, cx_labels, rotation='horizontal')

        plt.savefig(outdir + '/SkipFrames_online2D_calib.png')

    if run_skips.shape[0] > 0:
        plt.clf()

        x_val = run_skips['Run'].to_numpy(dtype='float16')
        y_val = run_skips['Gap'].to_numpy(dtype='float16')
        plt.scatter(x_val, y_val)
        plt.xticks(rx_runs, rx_labels, rotation='horizontal')

        plt.savefig(outdir + '/SkipFrames_online2D_run.png')


def make_composite_figure(good_runs, y_vals, x_vals, out_name=None, y_range=[-0.05, 1.05]):

        plt.clf()
        ncols = 3
        nrows = 2

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
                if ct == len(good_runs):
                    break

                # reset time values to 0 as onset
                x_val = np.array(x_vals[ct])
                x_0 = x_val[0]
                x_val -= x_0

                y_val = y_vals[ct]
                #m, b = np.polyfit(x_val, y_val, 1)
                #p_of_x = m*(x_val) + b
                p3, p2, p1, p0 = np.polyfit(x_val, y_val, 3)
                p_of_x = p3*(x_val**3) + p2*(x_val**2) + p1*(x_val) + p0
                #p6, p5, p4, p3, p2, p1, p0 = np.polyfit(x_val, y_val, 6)
                #p_of_x = p6*(x_val**6) + p5*(x_val**5) + p4*(x_val**4) + p3*(x_val**3) + p2*(x_val**2) + p1*(x_val) + p0

                ax[r][c].plot(x_val, y_val)
                ax[r][c].plot(x_val, p_of_x)
                ax[r][c].set_ylim(y_range)

                ax[r][c].set_title('Run {}'.format(good_runs[ct]), pad=30)
                ct += 1

                #ax[r][c].spines['top'].set_visible(False)
                #ax[r][c].spines['right'].set_visible(False)

        fig.add_subplot(111, frameon=False)

        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time (s)", labelpad=20)
        plt.ylabel("Gaze position", labelpad=20)

        fig.tight_layout(pad=6.0)

        if out_name is not None:
            plt.savefig(out_name)


def make_gaze_figures(good_runs, calib_gaze_allruns, run_gaze_allruns, out_name=None):
    '''
    E.g.,
    good_runs = ["1", "2", "6"]
    calib_gaze_allruns = [(cg_x1, cg_y1, cg_times1), (cg_x2, cg_y2, cg_times2), (cg_x6, cg_y6, cg_times6)]
    run_gaze_allruns = [(cg_x1, cg_y1, cg_times1), (cg_x2, cg_y2, cg_times2), (cg_x6, cg_y6, cg_times6)]
    '''

    # rearrange data to make plots
    c_times = []
    c_x = []
    c_y = []
    r_times = []
    r_x = []
    r_y = []

    for i in range(len(good_runs)):
        cgaze_x, cgaze_y, c_time = calib_gaze_allruns[i]
        c_times.append(c_time)
        c_x.append(cgaze_x)
        c_y.append(cgaze_y)

        rgaze_x, rgaze_y, r_time = run_gaze_allruns[i]
        r_times.append(r_time)
        r_x.append(rgaze_x)
        r_y.append(rgaze_y)

    make_composite_figure(good_runs, c_x, c_times, out_name=out_name + '_Xposition_calib.png')
    make_composite_figure(good_runs, c_y, c_times, out_name=out_name + '_Yposition_calib.png')

    make_composite_figure(good_runs, r_x, r_times, out_name=out_name + '_Xposition_run.png')
    make_composite_figure(good_runs, r_y, r_times, out_name=out_name + '_Yposition_run.png')


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


def assess_timegaps(run, t_stamps, threshold = 0.004016):
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
            skip_idx.append([int(run), i-1, diff, t_stamps[i-1]])

    return time_diff, skip_idx


def qc_report_summary(list_data, output_path, output_name, data_type, cf_thresh=0.6, tg_thresh = 0.004016):
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

    #time_diff, skip_idx = assess_timegaps(t_stamps, tg_thresh)

    x = np.array(positions)[:, 0].tolist()
    y = np.array(positions)[:, 1].tolist()
    times = np.array(positions)[:, 2].tolist()
    #_, _, _ = export_line_plot(confidences, os.path.join(output_path, 'confidence_' + output_name + '.png'), y_range=[-0.05, 1.05])
    #export_line_plot(time_diff, output_name + '_timediff.png')

    if data_type == 'gaze':
        #xdiff, x_m, x_b = export_line_plot(x, os.path.join(output_path, 'Xposition_' + output_name + '.png'), times, mid_val=0.5, y_range=[-0.05, 1.05])
        #ydiff, y_m, y_b = export_line_plot(y, os.path.join(output_path, 'Yposition_' + output_name + '.png'), times, mid_val=0.5, y_range=[-0.05, 1.05])
        xdiff, x_m, x_b = export_line_plot(x, None, times, mid_val=0.5, y_range=[-0.05, 1.05])
        ydiff, y_m, y_b = export_line_plot(y, None, times, mid_val=0.5, y_range=[-0.05, 1.05])

    elif data_type == 'pupils':
        #xdiff, x_m, x_b = export_line_plot(x, os.path.join(output_path, 'Xposition_' + output_name + '.png'), times, mid_val=320, y_range=[-4.0, 644.0])
        #ydiff, y_m, y_b = export_line_plot(y, os.path.join(output_path, 'Yposition_' + output_name + '.png'), times, mid_val=240, y_range=[-4.0, 484.0])
        xdiff, x_m, x_b = export_line_plot(x, None, times, mid_val=320, y_range=[-4.0, 644.0])
        ydiff, y_m, y_b = export_line_plot(y, None, times, mid_val=240, y_range=[-4.0, 484.0])

    '''
    if len(skip_idx) > 0:
        print(os.path.basename(output_name) + ' has ' + str(len(skip_idx)) + ' time gaps')
        np.savez(output_name + '_QCrep.npz', confidence = np.array(confidences), position = np.array(positions), time_diff = np.array(time_diff), time_gaps = np.array(skip_idx))
    else:
        np.savez(output_name + '_QCrep.npz', confidence = np.array(confidences), position = np.array(positions), time_diff = np.array(time_diff))
    '''
    return (x, y, times), d1, (xdiff, ydiff, x_m, x_b, y_m, y_b)



def process_run(cfg, run):

    gct = str(cfg['gaze_confidence_threshold'])
    pct = str(cfg['pupil_confidence_threshold'])

    gaze_report = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + gct + ' Confidence Threshold', 'Outside Screen Area', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept'])
    pupil_report = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + pct + ' Confidence Threshold', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept'])

    # check for missing frames in calibration eye movie (mp4)
    g_pool = make_detection_gpool()
    calib_eye_file = File_Source(g_pool, source_path=cfg['run' + run + '_calib_mp4'])
    calib_t_stamps = calib_eye_file.timestamps
    diff_list, gap_idx = assess_timegaps(run, calib_t_stamps, cfg['time_threshold'])
    if len(gap_idx) > 0:
        #export as .tsv
        np.savetxt(os.path.join(cfg['out_dir'], 'qc', 'run' + run + '_calib_framegaps.tsv'), np.array(gap_idx), delimiter="\t")

    # check for missing frames in main run eye movie (mp4)
    g_pool = make_detection_gpool()
    run_eye_file = File_Source(g_pool, source_path=cfg['run' + run + '_run_mp4'])
    run_t_stamps = run_eye_file.timestamps
    diff_list, gap_idx = assess_timegaps(run, run_t_stamps, cfg['time_threshold'])
    if len(gap_idx) > 0:
        #export as .tsv
        np.savetxt(os.path.join(cfg['out_dir'], 'qc', 'run' + run + '_run_framegaps.tsv'), np.array(gap_idx), delimiter="\t")

    # QC online pupils from calibration sequence
    calib_online_pupils = load_pldata_file(cfg['run' + run + '_calib_mp4'][:-9], 'pupil')[0]
    (cp_x, cp_y, cp_times), cp_on2d_d, (xdiff, ydiff, x_m, x_b, y_m, y_b) = qc_report_summary(calib_online_pupils, cfg['out_dir'] + '/qc', 'pupil_calib_online2D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
    cp_on2d_d = [cp_on2d_d[0], 'Calib', 'Online2D', 'Run' + run, cp_on2d_d[1], xdiff, ydiff, x_m, x_b, y_m, y_b]
    pupil_report = pupil_report.append(pd.Series(cp_on2d_d, index=pupil_report.columns), ignore_index=True)

    # QC online gaze from calibration sequence
    calib_online_gaze = load_pldata_file(cfg['run' + run + '_calib_mp4'][:-9], 'gaze')[0]
    (cg_x, cg_y, cg_times), cg_on2d_d, (xdiff, ydiff, x_m, x_b, y_m, y_b) = qc_report_summary(calib_online_gaze, cfg['out_dir'] + '/qc', 'gaze_calib_online2D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
    cg_on2d_d = [cg_on2d_d[0], 'Calib', 'Online2D', 'Run' + run, cg_on2d_d[1], cg_on2d_d[2], xdiff, ydiff, x_m, x_b, y_m, y_b]
    gaze_report = gaze_report.append(pd.Series(cg_on2d_d, index=gaze_report.columns), ignore_index=True)

    # QC online pupils from main run (task)
    run_online_pupils = load_pldata_file(cfg['run' + run + '_run_mp4'][:-9], 'pupil')[0]
    (rp_x, rp_y, rp_times), rp_on2d_d, (xdiff, ydiff, x_m, x_b, y_m, y_b) = qc_report_summary(run_online_pupils, cfg['out_dir'] + '/qc', 'pupil_run_online2D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
    rp_on2d_d = [rp_on2d_d[0], 'Run', 'Online2D', 'Run' + run, rp_on2d_d[1], xdiff, ydiff, x_m, x_b, y_m, y_b]
    pupil_report = pupil_report.append(pd.Series(rp_on2d_d, index=pupil_report.columns), ignore_index=True)

    # QC online gaze from main run (task)
    run_online_gaze = load_pldata_file(cfg['run' + run + '_run_mp4'][:-9], 'gaze')[0]
    (rg_x, rg_y, rg_times), rg_on2d_d, (xdiff, ydiff, x_m, x_b, y_m, y_b) = qc_report_summary(run_online_gaze, cfg['out_dir'] + '/qc', 'gaze_run_online2D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
    rg_on2d_d = [rg_on2d_d[0], 'Run', 'Online2D', 'Run' + run, rg_on2d_d[1], rg_on2d_d[2], xdiff, ydiff, x_m, x_b, y_m, y_b]
    gaze_report = gaze_report.append(pd.Series(rg_on2d_d, index=gaze_report.columns), ignore_index=True)

    gaze_data = ((cg_x, cg_y, cg_times), (rg_x, rg_y, rg_times))

    return pupil_report, gaze_report, gaze_data



if __name__ == "__main__":
    '''
    Script performs quality check for online pupil and gaze outputs for all runs
    from a single THINGS session.

    Note that the same config file can be used to run offline_calibration_THINGS.py, which
    outputs offline pupil and gaze measures (in 2d and optionally in 3d)
    '''

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    pct = str(cfg['pupil_confidence_threshold'])
    gct = str(cfg['gaze_confidence_threshold'])

    pupil_reports = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + pct + ' Confidence Threshold', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept'])
    gaze_reports = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + gct + ' Confidence Threshold', 'Outside Screen Area', 'X diff from mid', 'Y diff from mid', 'X slope', 'X intercept', 'Y slope', 'Y intercept'])

    calib_gaze_allruns = []
    run_gaze_allruns = []
    good_runs = []

    for run in cfg['runs']:
        print('Run ' + str(run))
        try:
            pupil_report, gaze_report, gaze_data = process_run(cfg, run)
            pupil_reports = pd.concat((pupil_reports, pupil_report), ignore_index=True)
            gaze_reports = pd.concat((gaze_reports, gaze_report), ignore_index=True)
            calib_gaze_allruns.append(gaze_data[0])
            run_gaze_allruns.append(gaze_data[1])
            good_runs.append(run)
        except:
            print('Something went wrong processing run ' + run)

    pupil_reports.to_csv(cfg['out_dir'] +'/qc/' + cfg['subject'] + '_ses' + cfg['session'] + '_pupil_report.tsv', sep='\t', header=True, index=False)
    gaze_reports.to_csv(cfg['out_dir'] +'/qc/' + cfg['subject'] + '_ses' + cfg['session'] + '_gaze_report.tsv', sep='\t', header=True, index=False)

    # # TODO
    # 1. load files for skipped frames and create figure
    # 2. create figure of X and Y positions over time, one global one for calinbration and for run
    make_gaze_figures(good_runs, calib_gaze_allruns, run_gaze_allruns, cfg['out_dir'] + '/qc/Gaze_online2D')
    make_frameskip_figure(good_runs, cfg['out_dir'] + '/qc')
