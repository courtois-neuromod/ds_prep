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


# Export recentered gaze into movie (episode)
def drawgaze(clip,fx,fy,r_zone):
    """
    Returns a filter that will add a moving circle that corresponds to the gaze mapped onto
    the frames. The position of the circle at time t is
    defined by (fx(t), fy(t)), and the radius of the circle
    by ``r_zone``.
    Requires OpenCV for the circling.
    Automatically deals with the case where part of the image goes
    offscreen.
    """

    def fl(gf,t):

        im_orig = gf(t)
        im = np.copy(im_orig)

        #im.setflags(write=1)
        h,w,d = im.shape
        x,y = int(fx(t)),int(fy(t))
        x1,x2 = max(0,x-r_zone),min(x+r_zone,w)
        y1,y2 = max(0,y-r_zone),min(y+r_zone,h)
        region_size = y2-y1,x2-x1

        orig = im[y1:y2, x1:x2]
        circled = cv2.circle(orig, (r_zone, r_zone), r_zone, (155, 0, 155), -1,
                             lineType=cv2.CV_AA)

        im[y1:y2, x1:x2] = circled
        return im

    return clip.fl(fl)


def drawgaze_confidence(clip,fx,fy,r_zone, f_conf):
    """
    Returns a filter that will add a moving circle that corresponds to the gaze mapped onto
    the frames. The position of the circle at time t is
    defined by (fx(t), fy(t)), and the radius of the circle
    by ``r_zone``.
    Requires OpenCV for the circling.
    Automatically deals with the case where part of the image goes
    offscreen.
    """

    def fl(gf,t):

        im_orig = gf(t)
        im = np.copy(im_orig)

        confi_thres = f_conf(t)

        if confi_thres < 0.1:
            #im.setflags(write=1)
            h,w,d = im.shape
            x,y = int(fx(t)),int(fy(t))
            x1,x2 = max(0,x-r_zone),min(x+r_zone,w)
            y1,y2 = max(0,y-r_zone),min(y+r_zone,h)
            region_size = y2-y1,x2-x1

            orig = im[y1:y2, x1:x2]
            circled = cv2.circle(orig, (r_zone, r_zone), r_zone, (155, 0, 155), -1,
                                 lineType=cv2.CV_AA)

            im[y1:y2, x1:x2] = circled

        return im

    return clip.fl(fl)


def drawgaze_multiples(clip, et_list, is_dg, zone_list, shade_list, fps):
    """
    Adds mulitple gazes from eyetracking and/or deepgaze per frame
    """
    def add_dots(image, coord, r_size, shade, dg=False):
        h,w,d = image.shape

        if dg:
            #assert(np.argmax(np.array(coord)[:, 0]) == 0)
            mean_weight = np.mean(np.array(coord)[:, 0])

        for i in range(len(coord)):
            x_norm, y_norm = coord[i][1:]

            # convert normalized to pixel
            x = int(np.floor(x_norm*720))
            x = 719 if x > 719 else x
            x = 0 if x < 0 else x

            y = int(np.floor((1 - y_norm)*512)) - 16
            y = 479 if y > 479 else y
            y = 0 if y < 0 else y

            # Largest (highest salience deepgaze) gets different hue
            if dg and i == 0:
                dot_shade = (250, 250, 0)
            else:
                dot_shade = shade

            if dg:
                # scale proportionally to salience value
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


def get_arguments():

    parser = argparse.ArgumentParser(description='Apply drift correction on pupil gaze data during free viewing (Friends) based on DeepGaze_MR coordinates')
    parser.add_argument('--gaze', default='run_s2e04a_online_gaze2D.npz', type=str, help='absolute path to gaze file')
    parser.add_argument('--film', default='friends_s2e04a_copy.mkv', type=str, help='absolute path to film .mkv file')
    parser.add_argument('--deepgaze_file', default=None, type=str, help='absolute path to deepgaze .npz local maxima file; if None, gaze center of mass used instead')

    parser.add_argument('--xdeg', type=int, default=None, help='degree of polynomial to correct drift in x')
    parser.add_argument('--ydeg', type=int, default=None, help='degree of polynomial to correct drift in y')
    parser.add_argument('--fps', type=float, default=29.97, help='frames per second')

    parser.add_argument('--gaze_confthres', type=float, default=0.98, help='gaze confidence threshold')
    parser.add_argument('--export_plots', action='store_true', help='if true, script exports QC plots')
    parser.add_argument('--export_mp4', action='store_true', help='if true, script exports episode movie with superimposed corrected and uncorrected gaze')
    #parser.add_argument('--use_deepgaze', action='store_true', help='if true, gaze recentered based on deepgaze, else from own centers of mass')

    parser.add_argument('--out_path', type=str, default='./results', help='path to output directory')
    parser.add_argument('--outname', default='test', type=str, help='name of output movie')
    args = parser.parse_args()

    return args


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
        while time_vals[i] < (frame_num + 1) * frame_dur:
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


def apply_poly(ref_times, distances, degree, clean_times, all_times, anchors = 150):

    if degree == 1:
        p1, p0 = np.polyfit(ref_times[anchors:-anchors], distances[anchors:-anchors], 1)
        p_of_clean = p1*(clean_times) + p0
        p_of_all = p1*(all_times) + p0

    elif degree == 2:
        p2, p1, p0 = np.polyfit(ref_times[anchors:-anchors], distances[anchors:-anchors], 2)
        p_of_clean = p2*(clean_times**2) + p1*(clean_times) + p0
        p_of_all = p2*(all_times**2) + p1*(all_times) + p0

    elif degree == 3:
        p3, p2, p1, p0 = np.polyfit(ref_times[anchors:-anchors], distances[anchors:-anchors], 3)
        p_of_clean = p3*(clean_times**3) + p2*(clean_times**2) + p1*(clean_times) + p0
        p_of_all = p3*(all_times**3) + p2*(all_times**2) + p1*(all_times) + p0

    return p_of_clean, p_of_all


def make_QC_figs(d, out_path, outname):

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
    plt.plot(frame_times, d['dist_x']) # distance from deepgaze (frames w single local maximum)
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized distance from target position", labelpad=20)
    plt.title("Drift in x gaze position")
    plt.savefig(os.path.join(out_path, outname + '_X_drift.png'))

    plt.clf()
    plt.ylim([-1, 1])
    plt.plot(clean_times, long_gaps)
    plt.plot(frame_times, np.array(d['et_y']) - y_mass)  # eyetracking per frame
    plt.plot(frame_times, d['dist_y']) # distance from deepgaze (frames w single local maximum)
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized distance from target position", labelpad=20)
    plt.title("Drift in y gaze position")
    plt.savefig(os.path.join(out_path, outname + '_Y_drift.png'))

    plt.clf()
    plt.ylim([-1, 1])
    plt.plot(clean_times, np.array(d['clean_y']) - y_mass)
    plt.plot(frame_times, d['dist_y'])
    plt.plot(clean_times, d['p_of_clean_y'])
    plt.xlabel("Time (s)", labelpad=20)
    plt.ylabel("Normalized distance from target position", labelpad=20)
    plt.title("Polynomial drift correction in y")
    plt.savefig(os.path.join(out_path, outname + '_Y_polycorr.png'))

    plt.clf()
    plt.ylim([-1, 1])
    plt.plot(clean_times, np.array(d['clean_x']) - x_mass)
    plt.plot(frame_times, d['dist_x'])
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

    print('len gz', len(gz))

    # get metrics to realign timestamps of movie frames and eyetracking gaze
    frame_count, zero_idx = get_indices(film_path, gz, fps)
    print(frame_count, zero_idx)

    # extract above-threshold normalized coordinates and their realigned time stamps
    all_vals, clean_vals = get_norm_coord(gz, zero_idx, args.gaze_confthres)
    all_x, all_y, all_times, all_conf = all_vals
    print(len(all_x), len(all_y), len(all_times), len(all_conf))
    clean_x, clean_y, clean_times, clean_conf, long_gaps = clean_vals
    print(len(clean_x), len(clean_y), len(clean_times), len(clean_conf), len(long_gaps))
    # assign uncorrected gaze to frames
    uncorr_gazes = gaze_2_frame(frame_count, fps, clean_x, clean_y, clean_times)
    print(len(uncorr_gazes))

    # Load deepgaze coordinates (local maxima per frame) pre-extracted from salience maps
    # Deepgaze salience files too large to save and run locally, so local maxima are extracted on compute canada...
    use_deepgaze = False
    all_DGvals = None
    deepgaze_file = args.deepgaze_file

    if deepgaze_file is not None:
        use_deepgaze = True
        all_DGvals = np.load(deepgaze_file, allow_pickle=True)['deepgaze_vals']
        # sanity check
        print(len(all_DGvals))
        assert(len(all_DGvals) == frame_count)

    # compute difference between frame's mean gaze and deepgaze's estimated coordinates (or average position in x and y)
    x_mass = 0.5
    y_mass = 0.7
    (et_x, et_y), (dist_x, dist_y), (dg_x, dg_y), frame_nums = get_distances(frame_count, uncorr_gazes,
                                                                             all_DGvals, use_deepgaze,
                                                                             x_mass, y_mass)

    print(len(et_x), len(et_y), len(dist_x), len(dist_y), len(dg_x), len(dg_y), len(frame_nums))
    # attribute time to middle of the frame
    frame_times = (np.array(frame_nums) + 0.5) / fps

    clean_times_arr = np.array(clean_times)
    all_times_arr = np.array(all_times)

    specs = ''
    if args.xdeg is not None:
        deg_x = args.xdeg
        specs += '_deg' + str(deg_x) + 'x'
        p_of_clean_x, p_of_all_x = apply_poly(frame_times, dist_x, deg_x,
                                              clean_times_arr, all_times_arr,
                                              anchors=150)
        all_x_aligned = np.array(all_x) - (p_of_all_x)
        clean_x_aligned = np.array(clean_x) - (p_of_clean_x)
    else:
        all_x_aligned = all_x
        clean_x_aligned = clean_x

    if args.ydeg is not None:
        deg_y = args.ydeg
        specs += '_deg' + str(deg_y) + 'y'
        p_of_clean_y, p_of_all_y = apply_poly(frame_times, dist_y, deg_y,
                                              clean_times_arr, all_times_arr,
                                              anchors=150)
        all_y_aligned = np.array(all_y) - (p_of_all_y)
        clean_y_aligned = np.array(clean_y) - (p_of_clean_y)
    else:
        all_y_aligned = all_y
        clean_y_aligned = clean_y

    if use_deepgaze:
        specs += '_DG'

    # Export corrected gaze as array of dictionaries
    all_gazes_array = np.empty(len(all_times), dtype='object')
    for i in range(len(all_gazes_array)):
        gaze = {}
        gaze['confidence'] = all_conf[i]
        gaze['norm_pos'] = (all_x[i], all_y[i])
        gaze['norm_pos_corr'] = (all_x_aligned[i], all_y_aligned[i])
        gaze['timestamp'] = all_times[i]
        all_gazes_array[i] =  gaze
    np.savez(os.path.join(args.out_path, 'All_gazecorr_' + args.outname + specs + '.npz'), gaze2d_corr = all_gazes_array)

    clean_gazes_array = np.empty(len(clean_times), dtype='object')
    for i in range(len(clean_times)):
        gaze = {}
        #gaze['confidence'] = clean_conf[i]
        gaze['norm_pos'] = (clean_x[i], clean_y[i])
        gaze['norm_pos_corr'] = (clean_x_aligned[i], clean_y_aligned[i])
        gaze['timestamp'] = clean_times[i]
        clean_gazes_array[i] =  gaze
    np.savez(os.path.join(args.out_path, 'Clean_gazecorr_' + args.outname + specs + '.npz'), gaze2d_corr = clean_gazes_array)


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


    if args.export_mp4:
        # assign corrected gaze to frames
        corr_gazes = gaze_2_frame(frame_count, fps, clean_x_aligned, clean_y_aligned, clean_times)

        clip = VideoFileClip(film_path)
        clip_gaze = clip.fx( drawgaze_multiples, [uncorr_gazes, corr_gazes], [False, False], [5, 5], [(155, 0, 0), (0, 0, 100)], fps)
        clip_gaze.write_videofile(os.path.join(args.out_path, args.outname + specs + '.mp4'))


if __name__ == '__main__':
    sys.exit(main())
