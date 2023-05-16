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
#from scipy.signal import savgol_filter
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
and time stamps. Only include high confidence gaze (specify cutoff threshold)

- Extract deepgaze coordinates from movie frames for which a single local maximum was identified (high confidence)

- For those frames (one deepgaze maximum) calculate the difference between the deepgaze coordinate and the average gaze position in x and y

- Draw a polynomial through the distribution of that difference over time

- Correct gaze position by removing the "difference" polynomial from the gaze position

- Option: Export plots and a movie (gaze over movie frames) to assess quality of drift correction / gaze mapping

- Alternative: correction without deepgaze. Instead of using difference from deepgaze to correct drift, compute the
difference between gaze and the "average" position within the frame.
0.5 in x = middle
0.7 in y is the height at which most faces are, most of the time. Friends is very consistent in its cinematography across episodes

'''


'''
UTILITY FUNCTIONS

Notes on frame dimentions:
Full screen is 1280:1024
Friends is projected onto 4:3 using the screen's entire width,
It is centered along the height dimension w padding along height (above and below movie) -> 1280:960
Friends frame sizes = 720:480 pixels (file dims) stretched into 1280:960 (on screen)
'''

def norm2pix(x_norms, y_norms):
    '''
    Function norm2pix converts gaze from normalized space [0, 1] to pixel space.
    From cartesian coordinates x and y as percent of screen -> coordinates h, w
    on image frame (array)) where 1 = 100% of screen size
    Input:
        x_norms (numpy array of floats): gaze coordinate in x (width)
        y_norms (numpy array of floats): gaze coordinate in y (height)
        0, 0 = bottom left (cartesian coordinates)
    Output:
        w_pix (list of int): gaze coordinate in pixel space (width)
        h_pix (list of int): gaze coordinate in pixel space (height)
    '''
    assert len(x_norms)==len(y_norms)

    screen_size = (1280, 1024)
    img_shape = (720, 480)

    # x (width) is projected along full width of the screen, so 100% = 720
    w_pix = np.floor(x_norms*720).astype(int)
    w_pix[w_pix > 719] = 719
    w_pix[w_pix < 0] = 0

    # y (height) is projected along 960/1024 of the screen height, so 93.75% = 480 and 100% = 512 (480*1024/960)
    # also, y coordinate is flipped to project onto stimulus (movie frame)
    h_pix = np.floor((1 - y_norms)*512).astype(int)
    h_pix -= 16 # remove padding to realign gaze onto unpadded frames

    h_pix[h_pix > 479] = 479
    h_pix[h_pix < 0] = 0

    return w_pix.tolist(), h_pix.tolist()


def pix2norm(h_pix, w_pix):
    '''
    Function pix2norm converts gaze coordinates from pixel space to normalized space
    coordinates h, w on image frame (array) -> cartesian coordinates x and y as percent of shown screen
    '''
    assert len(h_pix)==len(w_pix)

    screen_size = (1280, 1024)
    img_shape = (720, 480)

    # x (width) is projected along full width of the screen, so 720 = 100%
    x_norm = np.array(w_pix / 720).astype(float)

    # y (height) is projected along 960/1024 of the screen height, so 93.75% = 480 and 100% = 512 (480*1024/960)
    # also, y coordinate is flipped to project onto stimulus (movie frame)
    y_norm = np.array(((480 - h_pix) + 16) / 512).astype(float) # 16 pixels reflects the padding around height

    return list(zip(x_norm.tolist(), y_norm.tolist()))


def get_arguments():

    parser = argparse.ArgumentParser(description='Apply drift correction on pupil gaze data during free viewing (Friends) based on DeepGaze_MR coordinates')
    parser.add_argument('--gaze', default='run_s2e04a_online_gaze2D.npz', type=str, help='absolute path to gaze file')
    parser.add_argument('--film', default='friends_s2e04a_copy.mkv', type=str, help='absolute path to film .mkv file')
    parser.add_argument('--deepgaze_file', default=None, type=str, help='absolute path to deepgaze .npz local maxima file; if None, gaze center of mass used instead')

    parser.add_argument('--xdeg', type=int, default=None, help='degree of polynomial to correct drift in x')
    parser.add_argument('--ydeg', type=int, default=None, help='degree of polynomial to correct drift in y')
    #parser.add_argument('--fps', type=float, default=29.97, help='frames per second')
    parser.add_argument('--gaze_confthres', type=float, default=0.98, help='gaze confidence threshold')

    parser.add_argument('--out_path', type=str, default='./results', help='path to output directory')
    parser.add_argument('--outname', default='test', type=str, help='name of output movie')
    args = parser.parse_args()

    return args


def get_indices(film_path, gz):

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
    fps = cap.get(cv2.CAP_PROP_FPS)
    lag = gz[-1]['timestamp'] - gz[0]['timestamp'] - frame_count/fps
    interval = lag - 1/fps

    zero_idx = 0
    while gz[zero_idx]['timestamp'] - gz[0]['timestamp'] < interval:
        zero_idx += 1

    if lag < 0:
        zero_idx = 109
    return frame_count, zero_idx, fps


def get_norm_coord(gz, zero_idx, conf_thresh=0.9, gap_thresh = 0.1):
    '''
    Realign gaze-movie timing based on very last eye frame
    Export normalized x and y coordinates, and their corresponding time stamp
    '''
    # build array of x, y and timestamps
    # all gaze values
    all_x = []
    all_y = []
    all_times = []
    all_conf = []

    # thresholded with gaze confidence
    clean_x = []
    clean_y = []
    clean_times = []
    clean_conf = []

    # if 0.0, no gap between current eye frame and previous, else 1.0 if gap > 0.1 s (should be 0.004 at 250 fps)
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

            if len(chunk) > chunk_thresh: # can technically set this threshold higher
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


def apply_poly(ref_times, distances, degree, all_times, anchors = 150):

    if degree == 1:
        p1, p0 = np.polyfit(ref_times[anchors:-anchors], distances[anchors:-anchors], 1)
        p_of_all = p1*(all_times) + p0

    elif degree == 2:
        p2, p1, p0 = np.polyfit(ref_times[anchors:-anchors], distances[anchors:-anchors], 2)
        p_of_all = p2*(all_times**2) + p1*(all_times) + p0

    elif degree == 3:
        p3, p2, p1, p0 = np.polyfit(ref_times[anchors:-anchors], distances[anchors:-anchors], 3)
        p_of_all = p3*(all_times**3) + p2*(all_times**2) + p1*(all_times) + p0

    elif degree == 4:
        p4, p3, p2, p1, p0 = np.polyfit(ref_times[anchors:-anchors], distances[anchors:-anchors], 4)
        p_of_all = p4*(all_times**4) + p3*(all_times**3) + p2*(all_times**2) + p1*(all_times) + p0

    return p_of_all


def main():

    args = get_arguments()

    film_path = args.film # e.g., '/home/labopb/Documents/Marie/neuromod/friends_eyetrack/video_stimuli/friends_s06e03a.mkv'
    #fps = args.fps # 29.97 frames per second for Friends
    gaze_file = args.gaze # e.g., '/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-03/ses-070/run_s6e03a_online_gaze2D.npz'
    gz = np.load(gaze_file, allow_pickle=True)['gaze2d']

    # get metrics to realign timestamps of movie frames and eyetracking gaze
    frame_count, zero_idx, fps = get_indices(film_path, gz, fps)

    # extract above-threshold normalized coordinates and their realigned time stamps
    all_vals, clean_vals = get_norm_coord(gz, zero_idx, args.gaze_confthres)
    all_x, all_y, all_times, all_conf = all_vals
    clean_x, clean_y, clean_times, clean_conf, long_gaps = clean_vals

    '''
    Correct by chunk centers of mass (no deepgaze): for reference only
    '''
    x_chunks, y_chunks, chunk_timepoints = get_centermass(clean_x, clean_y, clean_times)

    # assign uncorrected gaze to frames
    uncorr_gazes = gaze_2_frame(frame_count, fps, clean_x, clean_y, clean_times)

    # Load deepgaze coordinates (local maxima per frame) pre-extracted from salience maps
    # Deepgaze salience files too large to save and run locally, so local maxima are extracted on compute canada...

    deepgaze_file = args.deepgaze_file
    all_DGvals = np.load(deepgaze_file, allow_pickle=True)['deepgaze_vals']
    use_deepgaze=True
    # sanity check
    assert(len(all_DGvals) == frame_count)

    # compute difference between frame's mean gaze and deepgaze's estimated coordinates (or average position in x and y)
    x_mass = 0.5
    y_mass = 0.7
    (et_x, et_y), (dist_x, dist_y), (dg_x, dg_y), frame_nums = get_distances(frame_count, uncorr_gazes,
                                                                             all_DGvals, use_deepgaze,
                                                                             x_mass, y_mass)

    # attribute time to middle of the frame
    frame_times = (np.array(frame_nums) + 0.5) / fps
    clean_times_arr = np.array(clean_times)
    all_times_arr = np.array(all_times)

    specs = ''
    if args.xdeg is not None:
        deg_x = args.xdeg
        specs += '_deg' + str(deg_x) + 'x'
        p_of_clean_x = apply_poly(frame_times, dist_x, deg_x, clean_times_arr, anchors=150)
        clean_x_aligned = np.array(clean_x) - (p_of_clean_x)

        p_of_all_x = apply_poly(frame_times, dist_x, deg_x, all_times_arr, anchors=150)
        all_x_aligned = np.array(all_x) - (p_of_all_x)

        p_of_x_chunks = apply_poly(chunk_timepoints, x_chunks, deg_x, clean_times_arr, anchors=1)
        clean_x_chunkaligned = np.array(clean_x) - (p_of_x_chunks - x_mass)

    else:
        all_x_aligned = all_x
        clean_x_aligned = clean_x
        clean_x_chunkaligned = clean_x

    if args.ydeg is not None:
        deg_y = args.ydeg
        specs += '_deg' + str(deg_y) + 'y'
        p_of_clean_y = apply_poly(frame_times, dist_y, deg_y, clean_times_arr, anchors=150)
        clean_y_aligned = np.array(clean_y) - (p_of_clean_y)

        p_of_all_y = apply_poly(frame_times, dist_y, deg_y, all_times_arr, anchors=150)
        all_y_aligned = np.array(all_y) - (p_of_all_y)

        p_of_y_chunks = apply_poly(chunk_timepoints, y_chunks, deg_y, clean_times_arr, anchors=1)
        clean_y_chunkaligned = np.array(clean_y) - (p_of_y_chunks - y_mass)

    else:
        all_y_aligned = all_y
        clean_y_aligned = clean_y
        clean_y_chunkaligned = clean_y

    plt.clf()
    plt.ylim([-17.5, 17.5])
    plt.plot(clean_times_arr, (clean_x_chunkaligned - clean_x_aligned)*17.5, color='xkcd:blue')#*17.5)
    plt.plot([clean_times_arr[0], clean_times_arr[-1]], [0, 0], linestyle='dashed', color='xkcd:red')#*17.5)
    #plt.xlabel("Time (s)", labelpad=20)
    #plt.ylabel("Degrees of visual angle", labelpad=20)
    #plt.title("CoM minus DG drift correction in X")
    plt.savefig(os.path.join(args.out_path, args.outname + '_diff_CoM_DG_X_deg.png'))

    plt.clf()
    plt.ylim([-14, 14])
    plt.plot(clean_times_arr, (clean_y_chunkaligned - clean_y_aligned)*14, color='xkcd:blue')#*14)
    plt.plot([clean_times_arr[0], clean_times_arr[-1]], [0, 0], linestyle='dashed', color='xkcd:red')#*17.5)
    #plt.xlabel("Time (s)", labelpad=20)
    #plt.ylabel("Degrees of visual angle", labelpad=20)
    #plt.title("CoM minus DG drift correction in Y")
    plt.savefig(os.path.join(args.out_path, args.outname + '_diff_CoM_DG_Y_deg.png'))


if __name__ == '__main__':
    sys.exit(main())
