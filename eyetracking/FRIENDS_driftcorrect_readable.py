import os
import sys, glob

import cv2
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from skimage.transform import resize
from skimage.feature import peak_local_max

import argparse

'''
NOTES
Gaze files must be saved as deserialized .npz files.
Online gaze files (e.g., gaze.pldata file) must be converted to .npz files with
convert_serialGaze_2_npz.py script as preliminary step

Support DeepGaze files are saved under : e.g., /deepgaze_files/friends/s06e03a_locmax_highestthres.npz
Coordinates of local maxima extractred from DeepGaze_MR salience maps were extracted using this script:
https://github.com/courtois-neuromod/deepgaze_mr/blob/mariedev_narval/scripts/process_saliences/extract_localmax_friends.py


Steps:

- From online or offline gaze data (from pupil), extract normalized gaze positions
and their corresponding time stamps. Only include high confidence gaze (above cutoff threshold)

- Extract deepgaze coordinates from movie frames for which a single local maximum was identified ("high confidence")

- For frames with one DG max and above-threshold gaze, calculate distance between deepgaze coordinate and average gaze position in x and y

- Within a sliding window, keep distances within 0.6 stdev of the median (the gazes most likely close to deepgaze fixation)

- Draw a polynomial through the distribution of those filtered distances plotted over time

- Correct gaze position by removing the "difference" polynomial from the gaze position

- Option: Export plots and a movie (gaze over movie frames) to assess quality of drift correction / gaze mapping

- Alternative: correction without deepgaze. When no Deepgaze file is provided, compute the
difference between gaze and the expected "average" position within the frame. Correct drift based on deviations from those positions
0.5 in x = middle
0.7 in y is the height at which most faces are, most of the time. Friends is very consistent in its cinematography across episodes


Notes on frame dimentions:
Full screen is 1280:1024
Friends is projected onto 4:3 using the screen's entire width,
It is centered along the height dimension w padding along height (above and below movie) -> 1280:960
Friends frame sizes = 720:480 pixels (file dims) stretched into 1280:960 (on screen)
Normalized gaze brought back to frame (not screen) pixel space (w padding) is 720 by 512 (16 pix of padding above and below)
'''


def get_arguments():

    parser = argparse.ArgumentParser(description='Apply drift correction on pupil gaze data during free viewing (Friends) based on DeepGaze_MR coordinates')
    parser.add_argument('--gaze', default='run_s2e04a_online_gaze2D.npz', type=str, help='absolute path to gaze file')
    parser.add_argument('--film', default='friends_s2e04a_copy.mkv', type=str, help='absolute path to film .mkv file')
    parser.add_argument('--deepgaze_file', default=None, type=str, help='absolute path to deepgaze .npz local maxima file; if None, center of mass correction used instead')

    parser.add_argument('--xdeg', type=int, default=None, help='degree of polynomial to correct drift in x')
    parser.add_argument('--ydeg', type=int, default=None, help='degree of polynomial to correct drift in y')
    parser.add_argument('--fps', type=float, default=29.97, help='frames per second')

    parser.add_argument('--gaze_confthres', type=float, default=0.98, help='gaze confidence threshold')
    parser.add_argument('--export_plots', action='store_true', help='if true, script exports QC plots')
    parser.add_argument('--export_mp4', action='store_true', help='if true, script exports episode movie with superimposed corrected and uncorrected gaze')
    parser.add_argument('--chunk_centermass', action='store_true', help='if true, also perform the center of mass correction per chunk, without DeepGaze, and plots it on mp4 for comparison')
    parser.add_argument('--savgol', action='store_true', help='if true, use Savitzky-Golay filter to correct drift w DeepGaze and plot it onto mp4 for comparison')
    #parser.add_argument('--use_deepgaze', action='store_true', help='if true, gaze recentered based on deepgaze, else from own centers of mass')

    parser.add_argument('--out_path', type=str, default='./results', help='path to output directory')
    parser.add_argument('--outname', default='test', type=str, help='name of output movie')
    args = parser.parse_args()

    return args


# Export recentered gaze into movie (half episode)
def drawgaze_multiples(clip, et_list, is_dg, zone_list, shade_list, fps):
    """
    Adds mulitple gazes from eyetracking and/or deepgaze per frame
    """
    def add_dots(image, coord, r_size, shade, dg=False):
        h,w,d = image.shape

        if dg:
            # if multiple DGaze local maxima, scale their size based on salience value
            mean_weight = np.mean(np.array(coord)[:, 0])

        for i in range(len(coord)):
            x_norm, y_norm = coord[i][1:]

            # convert normalized gaze to pixel space
            # (frame is 720w by 480h, w padding along height to fit frame)
            x = int(np.floor(x_norm*720))
            x = 719 if x > 719 else x
            x = 0 if x < 0 else x

            y = int(np.floor((1 - y_norm)*512)) - 16
            y = 479 if y > 479 else y
            y = 0 if y < 0 else y

            # Largest (highest salience deepgaze) gets different hue (yellow)
            if dg and i == 0:
                dot_shade = (250, 250, 0)
            else:
                dot_shade = shade

            if dg:
                # scale DG local maxima proportionally to salience value
                r_zone = int(np.floor((r_size*coord[i][0])/mean_weight))
            else:
                r_zone = r_size

            x1,x2 = max(0,x-r_zone),min(x+r_zone,w)
            y1,y2 = max(0,y-r_zone),min(y+r_zone,h)
            region_size = y2-y1,x2-x1

            orig = image[y1:y2, x1:x2]

            circled = cv2.circle(orig, (r_zone, r_zone), r_zone, dot_shade, -1,
                                 lineType=cv2.CV_AA)

            image[y1:y2, x1:x2] = circled

        return image


    def fl(gf,t):

        im_orig = gf(t)
        im = np.copy(im_orig)

        indice = np.floor(t * fps).astype('int')

        for j in range(len(et_list)):

            et_file = et_list[j]

            if indice < len(et_file) and len(et_file[indice]) > 0:
                im = add_dots(im, et_file[indice], zone_list[j], shade_list[j], dg=is_dg[j])

        return im

    return clip.fl(fl)


def get_indices(film_path, gz, fps):

    '''
    Count number of frames in episode .mkv
    '''
    cap = cv2.VideoCapture(film_path)

    success = True
    frame_count = 0
    while success:
        success, image = cap.read()
        if success:
            frame_count += 1

    '''
    Determine onset time for first movie frame based on time of last eye frame
    Note: there is no logged time for movie frames in Friends dataset (unlike mario's bk2 files)

    The movie and eyetracking camera onsets are not perfectly aligned, but their offsets are very close (by ~1 frame).
    For greater timing, align last movie frame with last reccorded eye gaze,
    and work backward based on the movie frame count and fps to determine movie onset.

    NOTE: this technique WILL fail if the eyetracking data is incomplete (e.g., stops mid-run);
    In that case ("lag" value is negative and zero_idx remains == 0)
    In that scenario, a zero_idx value of around 110 is a reasonable approximation
    '''
    lag = gz[-1]['timestamp'] - gz[0]['timestamp'] - frame_count/fps
    interval = lag - 1/fps

    zero_idx = 0
    while gz[zero_idx]['timestamp'] - gz[0]['timestamp'] < interval:
        zero_idx += 1

    if lag < 0:
        zero_idx = 109
    return frame_count, zero_idx


def get_norm_coord(gz, zero_idx, conf_thresh=0.9, gap_thresh = 0.1):
    '''
    Realign gaze-movie timing based on very last eye frame
    Export normalized x and y coordinates, and their corresponding time stamp
    '''
    # build array of x, y and timestamps
    # all gaze values (unfiltered)
    all_x = []
    all_y = []
    all_times = []
    all_conf = []

    # gaze filtered based on confidence threshold
    clean_x = []
    clean_y = []
    clean_times = []
    clean_conf = []

    # if 0.0, no gap between current eye frame and previous, else 1.0 if gap > 0.1 s (should be 0.004 at 250 fps)
    # TODO: improve this metric (plotting isn't great)...
    long_gap = [0.0]

    for i in range(zero_idx, len(gz)):
        gaze = gz[i]
        x_norm, y_norm = gaze['norm_pos']
        cfd = gaze['confidence']

        timestp = gaze['timestamp'] - gz[zero_idx]['timestamp']
        all_x.append(x_norm)
        all_y.append(y_norm)
        all_times.append(timestp)
        all_conf.append(cfd)

        if cfd > conf_thresh:
            # Keep x, y coord values that fall outside the 0.0-1.0 (normalized) interval since might still be salvageable post drift correction...
            #timestp = gaze['timestamp'] - gz[0]['timestamp']
            clean_x.append(x_norm)
            clean_y.append(y_norm)
            clean_conf.append(cfd)
            if len(clean_times) > 0:
                if (timestp - clean_times[-1]) > gap_thresh: #0.1:
                    long_gap.append(1.0)
                else:
                    long_gap.append(0.0)
            clean_times.append(timestp)

    # Quick hack to prevent out-of-range interpolation issues by padding extremes
    # in gaze below-confidence thresh gates at either end of the run
    if clean_times[0] > 0:
        clean_times = [0.0] + clean_times
        long_gap = [0.0, 1.0] + long_gap[1:]
        clean_conf = [gz[zero_idx]['confidence']] + clean_conf
        x, y = gz[zero_idx]['norm_pos']
        clean_x = [x] + clean_x
        clean_y = [y] + clean_y

    if clean_times[-1] < gz[-1]['timestamp'] - gz[zero_idx]['timestamp']:
        clean_times.append(gz[-1]['timestamp'] - gz[zero_idx]['timestamp'])
        clean_conf.append(gz[-1]['confidence'])
        long_gap.append(1.0)
        x, y = gz[-1]['norm_pos']
        clean_x.append(x)
        clean_y.append(y)

    return (all_x, all_y, all_times, all_conf), (clean_x, clean_y, clean_times, clean_conf, long_gap)


def get_centermass(all_x, all_y, all_times):
    '''
    This is a "legacy" function, for comparison purposes (only plot on mp4):
    performs Center-of-Mass driftcorrection the way I used to validate DeepGaze
    Iterate through chunks of gaze of a certain duration
    Here, ever 20 seconds, sample 5 seconds worth of eye-tracking coordinates
    '''
    sample_dur = 5 # in seconds
    chunk_thresh = 5 # 250
    jump = 20 #5 #20

    idx = 0
    chunk_timepoints = []
    x_chunks = []
    y_chunks = []

    for j in range(len(all_times)):

        if all_times[j] > (idx * jump):
            i = 0
            chunk = []
            while i + j < len(all_times) and all_times[j+i] < ((idx * jump) + sample_dur):
                chunk.append([all_x[i+j], all_y[i+j]])
                i += 1

            tstamp = (idx * jump) + (sample_dur/2)

            if len(chunk) > chunk_thresh: # can probably set this threshold higher
                chunk = np.array(chunk)
                x_mean, y_mean = np.mean(chunk, axis=0)
                x_chunks.append(x_mean)
                y_chunks.append(y_mean)
                chunk_timepoints.append(tstamp)
            idx += 1

    return x_chunks, y_chunks, chunk_timepoints


def gaze_2_frame(frame_count, fps, x_vals, y_vals, time_vals):
    '''
    Assign gaze positions to movie frames
    Input:
        x_vals (list of float): x coordinates
        y_vals (list of float): y coordinates
        time_vals (list of float): gaze's timestamp
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


def get_distances(frame_count, gazes_perframe, DGvals_perframe, use_deepgaze=False, x_mass=0.5, y_mass=0.7):
    '''
    Calculates distances between mean gaze (averaged per frame) and DeepGaze
    If use_deepgaze is False, then calculates distances to target centers of mass x = 0.5 and y = 0.7
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
                # For spatial reference, ONLY use only "high confidence" Deepgaze points
                # for which a single local maximum was identified (>.8 max, >20 pixels of distance)
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


def fit_savgol(frame_times, dist_x, dist_y, all_times, anchor=50):
    '''
    Currently only for visualization purposes (to plot on movie mp4):
    Applies the Savitzky-Golay filter to a distance distribution in x and y,
    then interpolates points from that distribution for specific gaze timestamps
    '''
    div = 3
    polyorder = 2 #2

    window_length = int(len(dist_y[anchor:-anchor])/div)

    if window_length % 2 == 0:
        window_length += 1 # must be odd number

    p_of_x_sav = savgol_filter(np.array(dist_x[anchor:-anchor]), window_length, polyorder, deriv=0, delta=1.0, axis=- 1, mode='nearest')
    p_of_y_sav = savgol_filter(np.array(dist_y[anchor:-anchor]), window_length, polyorder, deriv=0, delta=1.0, axis=- 1, mode='nearest')

    f_x = interp1d(frame_times[anchor:-anchor], p_of_x_sav, fill_value='extrapolate')
    f_y = interp1d(frame_times[anchor:-anchor], p_of_y_sav, fill_value='extrapolate')

    p_of_x_sav_interp = f_x(all_times)
    p_of_y_sav_interp = f_y(all_times)

    return p_of_x_sav_interp, p_of_y_sav_interp


def make_QC_figs(d, out_path, outname):
    '''
    Export a bunch of quick and dirty figs to assess drift estimation goodness of fit
    '''
    clean_times = d['clean_times']
    all_times = d['all_times']
    frame_times = d['frame_times']
    long_gaps = d['long_gaps']
    x_mass = d['x_mass'] #0.5
    y_mass = d['y_mass'] #0.7

    plt.clf()
    plt.ylim([-1, 1])
    plt.plot(clean_times, long_gaps)
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Gaps of > 0.1s in confident gaze", labelpad=20)
    plt.title("Gaps in high confidence gaze")
    plt.savefig(os.path.join(out_path, outname + '_gazedata_gaps.png'))

    plt.clf()
    plt.ylim([-1, 1])
    plt.plot(clean_times, long_gaps)
    plt.plot(frame_times, np.array(d['et_x']) - x_mass)  # eyetracking per frame
    plt.scatter(frame_times, d['dist_x'], s=2) # distance from deepgaze (frames w single local maximum)
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized distance from target position", labelpad=20)
    plt.title("Drift in x gaze position")
    plt.savefig(os.path.join(out_path, outname + '_X_drift.png'))

    plt.clf()
    plt.ylim([-1, 1])
    plt.plot(clean_times, long_gaps)
    plt.plot(frame_times, np.array(d['et_y']) - y_mass)  # eyetracking per frame
    plt.scatter(frame_times, d['dist_y'], s=2) # distance from deepgaze (frames w single local maximum)
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized distance from target position", labelpad=20)
    plt.title("Drift in y gaze position")
    plt.savefig(os.path.join(out_path, outname + '_Y_drift.png'))

    plt.clf()
    plt.ylim([-1, 1])
    plt.plot(clean_times, np.array(d['clean_y']) - y_mass)
    plt.scatter(frame_times, d['dist_y'], s=2)
    plt.plot(clean_times, d['p_of_clean_y'])
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized distance from target position", labelpad=20)
    plt.title("Polynomial drift correction in y")
    plt.savefig(os.path.join(out_path, outname + '_Y_polycorr.png'))

    plt.clf()
    plt.ylim([-1, 1])
    plt.plot(clean_times, np.array(d['clean_x']) - x_mass)
    plt.scatter(frame_times, d['dist_x'], s=2)
    plt.plot(clean_times, d['p_of_clean_x'])
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized distance from target position", labelpad=20)
    plt.title("Polynomial drift correction in x")
    plt.savefig(os.path.join(out_path, outname + '_X_polycorr.png'))

    plt.clf()
    plt.ylim([-3, 3])
    plt.plot(clean_times, d['clean_y'])
    plt.plot(clean_times, d['clean_y_aligned'])
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized gaze position", labelpad=20)
    plt.title("Gaze position in y before and after correction")
    plt.savefig(os.path.join(out_path, outname + '_Y_corrected.png'))

    plt.clf()
    plt.ylim([-3, 3])
    plt.plot(clean_times, d['clean_x'])
    plt.plot(clean_times, d['clean_x_aligned'])
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized gaze position", labelpad=20)
    plt.title("Gaze position in x before and after correction")
    plt.savefig(os.path.join(out_path, outname + '_X_corrected.png'))


def main():

    args = get_arguments()

    film_path = args.film # e.g., '/home/labopb/Documents/Marie/neuromod/friends_eyetrack/video_stimuli/friends_s06e03a.mkv'
    fps = args.fps # 29.97 frames per second for Friends
    gaze_file = args.gaze # e.g., '/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-03/ses-070/run_s6e03a_online_gaze2D.npz'
    gz = np.load(gaze_file, allow_pickle=True)['gaze2d']

    # get metrics to realign timestamps of movie frames and eyetracking gaze
    frame_count, zero_idx = get_indices(film_path, gz, fps)

    # extract above-threshold normalized coordinates and their realigned time stamps
    all_vals, clean_vals = get_norm_coord(gz, zero_idx, args.gaze_confthres)
    # all gazes (unfiltered): to export entire dataset?
    all_x, all_y, all_times, all_conf = all_vals
    # filtered gazes (above confidence threshold)
    clean_x, clean_y, clean_times, clean_conf, long_gaps = clean_vals

    '''
    For reference only (legacy?)
    Correct drift with centers of mass from chunks sampled at set intervals (independant from deepgaze):
    NOTE: this is the non-deepgaze based approach, strictly for comparison (to plot on movies if exported)
    '''
    if args.chunk_centermass:
        x_chunks, y_chunks, chunk_timepoints = get_centermass(clean_x, clean_y, clean_times)

    # assign uncorrected gaze to movie frames (to line-up w deepgaze)
    uncorr_gazes = gaze_2_frame(frame_count, fps, clean_x, clean_y, clean_times)

    # Load deepgaze coordinates (local maxima per frame) pre-extracted from salience maps
    # Deepgaze salience files too large to save and run locally, so local maxima are extracted on compute canada...
    # for each frame, script exctracts local maxima salience and coordinates from deepgaze salience maps (at set threshold)
    use_deepgaze = False
    all_DGvals = None
    deepgaze_file = args.deepgaze_file

    # If a deepgaze file is provided, drift is corrected in relation to deepgaze frames w a single locl maximum,
    # otherwise it uses centers of mass and expected mean gaze position in X and Y
    if deepgaze_file is not None:
        use_deepgaze = True
        all_DGvals = np.load(deepgaze_file, allow_pickle=True)['deepgaze_vals']
        # sanity check
        assert(len(all_DGvals) == frame_count)

    # compute normalized distance between movie frame's mean gaze and deepgaze's estimated coordinates
    # (or expected average position in x and y (x_mass and y_mass) if no deepgaze file is given)
    x_mass = 0.5
    y_mass = 0.7
    (et_x, et_y), (dist_x, dist_y), (dg_x, dg_y), frame_nums = get_distances(frame_count, uncorr_gazes,
                                                                             all_DGvals, use_deepgaze,
                                                                             x_mass, y_mass)
    # attribute time stamp in s to middle of each movie frame
    frame_times = (np.array(frame_nums) + 0.5) / fps
    clean_times_arr = np.array(clean_times)
    all_times_arr = np.array(all_times)

    anchors = [150,150] # to exclude values at extremities (when black screen, sometimes participants look off the screen while deepgaze defaults to middle...)
    if use_deepgaze:
        # remove distances too far from median within sliding windows for cleaner signal
        frame_times, et_x, et_y, dist_x, dist_y = median_clean(frame_times, et_x, et_y, dist_x, dist_y)
        anchors = [50,50]

    specs = ''
    if args.xdeg is not None:
        deg_x = args.xdeg
        specs += '_deg' + str(deg_x) + 'x'
        # to apply correction for filtered gaze above confidence threshold
        p_of_clean_x = apply_poly(frame_times, dist_x, deg_x, clean_times_arr, anchors=anchors)
        clean_x_aligned = np.array(clean_x) - (p_of_clean_x)

        # to apply correction to all gaze (no confidence threshold applied)
        p_of_all_x = apply_poly(frame_times, dist_x, deg_x, all_times_arr, anchors=anchors)
        all_x_aligned = np.array(all_x) - (p_of_all_x)

        if args.chunk_centermass:
            # additional option for reference only: contrast center of mass correction to deepgaze
            p_of_x_chunks = apply_poly(chunk_timepoints, x_chunks, deg_x, clean_times_arr, anchors=[1, 1])
            clean_x_chunkaligned = np.array(clean_x) - (p_of_x_chunks - x_mass)

    else:
        # don't correct in X if no polynomial degree given
        all_x_aligned = all_x
        clean_x_aligned = clean_x
        if args.chunk_centermass:
            clean_x_chunkaligned = clean_x

    if args.ydeg is not None:
        deg_y = args.ydeg
        specs += '_deg' + str(deg_y) + 'y'
        # correct filtered gaze
        p_of_clean_y = apply_poly(frame_times, dist_y, deg_y, clean_times_arr, anchors=anchors)
        clean_y_aligned = np.array(clean_y) - (p_of_clean_y)

        # correct all gaze
        p_of_all_y = apply_poly(frame_times, dist_y, deg_y, all_times_arr, anchors=anchors)
        all_y_aligned = np.array(all_y) - (p_of_all_y)

        if args.chunk_centermass:
            p_of_y_chunks = apply_poly(chunk_timepoints, y_chunks, deg_y, clean_times_arr, anchors=[1, 1])
            clean_y_chunkaligned = np.array(clean_y) - (p_of_y_chunks - y_mass)

    else:
        all_y_aligned = all_y
        clean_y_aligned = clean_y
        if args.chunk_centermass:
            clean_y_chunkaligned = clean_y

    if use_deepgaze:
        specs += '_DG'

    if args.savgol:
        # for comparison only (shown on mp4): apply salgov algo to distance between deepgaze and tracked gaze instead of polynomial
        p_of_x_svg, p_of_y_svg = fit_savgol(frame_times, dist_x, dist_y, clean_times, anchors)
        clean_x_svgaligned = np.array(clean_x) - (p_of_x_svg)
        clean_y_svgaligned = np.array(clean_y) - (p_of_y_svg)

    # Export corrected gaze as array of dictionaries (all gaze)
    all_gazes_array = np.empty(len(all_times), dtype='object')
    for i in range(len(all_gazes_array)):
        gaze = {}
        gaze['confidence'] = all_conf[i]
        gaze['norm_pos'] = (all_x[i], all_y[i])
        gaze['norm_pos_corr'] = (all_x_aligned[i], all_y_aligned[i])
        gaze['timestamp'] = all_times[i]
        all_gazes_array[i] =  gaze
    np.savez(os.path.join(args.out_path, 'All_gazecorr_' + args.outname + specs + '.npz'), gaze2d_corr = all_gazes_array)

    # Export corrected gaze as array of dictionaries (filtered gaze above confidence threshold)
    clean_gazes_array = np.empty(len(clean_times), dtype='object')
    for i in range(len(clean_times)):
        gaze = {}
        #gaze['confidence'] = clean_conf[i]
        gaze['norm_pos'] = (clean_x[i], clean_y[i])
        gaze['norm_pos_corr'] = (clean_x_aligned[i], clean_y_aligned[i])
        gaze['timestamp'] = clean_times[i]
        clean_gazes_array[i] =  gaze
    np.savez(os.path.join(args.out_path, 'Clean_gazecorr_' + args.outname + specs + '.npz'), gaze2d_corr = clean_gazes_array)

    # Option to export plots to assess correction fit (for QC)
    if args.export_plots:
        plot_data_dict = {}
        plot_data_dict['all_times'] = all_times
        plot_data_dict['clean_times'] = clean_times
        plot_data_dict['frame_times'] = frame_times
        plot_data_dict['long_gaps'] = long_gaps

        plot_data_dict['clean_x'] = clean_x
        plot_data_dict['clean_y'] = clean_y
        plot_data_dict['et_x'] = et_x
        plot_data_dict['et_y'] = et_y
        plot_data_dict['x_mass'] = x_mass
        plot_data_dict['y_mass'] = y_mass
        plot_data_dict['dist_x'] = dist_x
        plot_data_dict['dist_y'] = dist_y
        plot_data_dict['p_of_clean_x'] = p_of_clean_x
        plot_data_dict['p_of_clean_y'] = p_of_clean_y
        plot_data_dict['clean_x_aligned'] = clean_x_aligned
        plot_data_dict['clean_y_aligned'] = clean_y_aligned

        make_QC_figs(plot_data_dict, args.out_path, args.outname)

    # create movie of episodes w uncorrected and corrected gaze super-imposed
    if args.export_mp4:
        # assign corrected gaze to frames
        corr_gazes = gaze_2_frame(frame_count, fps, clean_x_aligned, clean_y_aligned, clean_times)

        clip = VideoFileClip(film_path)

        if args.chunk_centermass:
            # option to add gaze corrected with center-of-mass approach
            corr_gazes_chunks = gaze_2_frame(frame_count, fps, clean_x_chunkaligned, clean_y_chunkaligned, clean_times)
            specs += '_CMass'
            if args.savgol:
                # option to add gaze corrected with savgol algo applied to gaze-deepgaze distances, instead of polynomial
                specs += '_SVG'
                corr_gazes_svg = gaze_2_frame(frame_count, fps, clean_x_svgaligned, clean_y_svgaligned, clean_times)
                clip_gaze = clip.fx( drawgaze_multiples, [uncorr_gazes, corr_gazes, corr_gazes_svg, corr_gazes_chunks], [False, False, False, False], [5, 5, 5, 5], [(155, 0, 0), (0, 0, 155), (0, 200, 0), (202, 228, 241)], fps)
            else:
                clip_gaze = clip.fx( drawgaze_multiples, [uncorr_gazes, corr_gazes, corr_gazes_chunks], [False, False, False], [5, 5, 5], [(155, 0, 0), (0, 0, 155), (202, 228, 241)], fps)

        else:
            # basic option: uncorrected and corrected (deepgaze) gaze
            clip_gaze = clip.fx( drawgaze_multiples, [uncorr_gazes, corr_gazes], [False, False], [5, 5], [(155, 0, 0), (0, 0, 155)], fps)
        clip_gaze.write_videofile(os.path.join(args.out_path, args.outname + specs + '.mp4'))


if __name__ == '__main__':
    sys.exit(main())
