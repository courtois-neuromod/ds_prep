import os
from moviepy.editor import *
from scipy.interpolate import interp1d
import cv2
import pandas as pd
import numpy as np

'''
Script overlaps gaze stimuli onto movie frames and exports as .mp4

NOTE: online gaze files (e.g., gaze.pldata) must be deserialized and
converted to .npz file with the convert_serialGaze_2_npz.py script

Frames per second (fps) for Friends is 29.97

Links for Documentation
https://github.com/Zulko/moviepy/blob/master/moviepy/video/tools/tracking.py
https://pypi.org/project/moviepy/
https://zulko.github.io/moviepy/getting_started/quick_presentation.html
https://zulko.github.io/moviepy/advanced_tools/advanced_tools.html#advancedtools

https://zulko.github.io/moviepy/examples/headblur.html
https://zulko.github.io/moviepy/_modules/moviepy/video/tools/drawing.html#circle

CV2 circle:
cv2.circle() method is used to draw a circle on any image
https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
'''


def get_arguments():

    parser = argparse.ArgumentParser(description='Perform off-line gaze mapping with 2D and 3D pupil detectors ')
    parser.add_argument('--gaze', default='', type=str, help='absolute path to gaze file')
    parser.add_argument('--config', default='config.json', type=str, help='absolute path to film .mkv file')
    parser.add_argument('--out_path', type=str, default='./results', help='path to output directory')
    parser.add_argument('--fps', type=float, default=29.97', help='frames per second')
    parser.add_argument('--gaze_confthres', type=float, default=0.98', help='gaze confidence threshold')
    parser.add_argument('--deepgaze', action='store_true', help='if true, map gaze from deepgaze coordinates')
    parser.add_argument('--partial', action='store_true', help='if true, only process a portion of the movie')
    parser.add_argument('--start_frame', type=int, default=0', help='first frame included in the segment')
    parser.add_argument('--num_frames', type=int, default=1000', help='number of frames included in the segment')
    parser.add_argument('--outname', default='test', type=str, help='name of output movie')
    args = parser.parse_args()

    return args


def norm2pix(x_norms, y_norms):
    '''
    Function converts gaze from normalized to pixel space
    Cartesian coordinates x and y as percent of screen -> coordinates h, w on image frame (array))

    Full screen is 1280:1024
    Friends is projected onto 4:3 using the screen's entire width,
    It is centered along the height dimension w padding above and below -> 1280:960

    Friends frame sizes = 720:480 pixels (file dims)
    Frames are projected onto the full screen (with stretching) with some padding along height (no padding along width)

    The padded height corresponds to 512 pixels
    480 -> 512 pixels (16 pixels above and below on height axis)

    Padded frames are then stretched into 1280:960 (on screen)
    '''
    assert len(x_norms)==len(y_norms)

    screen_size = (1280, 1024)
    img_shape = (720, 480)

    # x (width) is projected along full width of the screen, so 100% = 720
    x_pix = np.floor(x_norms*720).astype(int)
    x_pix[x_pix > 719] = 719
    x_pix[x_pix < 0] = 0

    # y (height) is projected along 960/1024 of the screen height, so 93.75% = 480 and 100% = 512 (480*1024/960)
    # also, y coordinate is FLIPPED to project onto stimulus (movie frame)
    y_pix = np.floor((1 - y_norms)*512).astype(int)
    y_pix -= 16 # remove padding to realign gaze onto unpadded frames

    y_pix[y_pix > 479] = 479
    y_pix[y_pix < 0] = 0

    return x_pix.tolist(), y_pix.tolist()


def get_eyetrack_function(gaze_file, d3=False, conf_thresh=0.98, time_info=(0.0, np.inf)):
    '''
    Function formats subject's gaze data (.npz file) to map onto movie frames
    NOTE that online gaze files (e.g., gaze.pldata) must be deserialized and
    converted to .npz file with the convert_serialGaze_2_npz.py script

    gaze_file (str): '/path/to/2d_gaze.npz'
    e.g., '/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-01/ses-039/2d_gaze.npz'
    d3 (bool): gaze is from 3d model
    conf_thresh (float): confidence threshold for gaze estimation
    '''

    run_gaze = np.load(gaze_file, allow_pickle=True)
    if d3:
        gz = run_gaze['gaze3d']
    else:
        gz = run_gaze['gaze2d']

    # Export normalized x and y coordinates, and their corresponding time stamp
    # (relative to the first eye frame capture = "time 0") for each gaze

    # build array of x, y and timestamps
    all_x = []
    all_y = []
    all_times = []

    start_time, end_time = time_info
    last_idx = 0
    #print(len(gz))
    for i in range(len(gz)):
        gaze = gz[i]

        timestp = gaze['timestamp'] - gz[0]['timestamp']
        # In case partial video doesn't start at 0
        if (timestp >= start_time) and (timestp <= end_time):
            last_idx = i
            if gaze['confidence'] > conf_thresh:
                x_norm, y_norm = gaze['norm_pos']

                all_x.append(x_norm)
                all_y.append(y_norm)
                all_times.append(timestp - start_time)

    # sometimes edge gazes have below-threshold confidence
    # the clunky code below prevents frame times from falling outisde interpolation range
    if all_times[0] > 0:
        all_times = [0.0] + all_times
        x, y = gz[2]['norm_pos']
        all_x = [x] + all_x
        all_y = [y] + all_y

    if all_times[-1] < gz[last_idx]['timestamp'] - gz[0]['timestamp'] - start_time:
        all_times.append(gz[last_idx]['timestamp'] - gz[0]['timestamp'] - start_time)
        x, y = gz[last_idx]['norm_pos']
        all_x.append(x_norm)
        all_y.append(y_norm)

    #print(str(100*(len(all_times)/len(gz))) + '% of gaze above confidence threshold')
    '''
    Note: normalized x and y are in cartesian space (x, y) where y is "flipped" (high value is high in the image)
    Image dims are in "pixel/matrix space": (h, w) where 0 = top and 1 is bottom for h; (0, 0) is top left corner
    Convert normalized coordinates into pixel space (unpadded)
    '''

    x_pix, y_pix = norm2pix(np.array(all_x), np.array(all_y))

    '''
    Interpolation fonctions
    https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
    'linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    '''
    f_x = interp1d(all_times, x_pix, kind='linear')
    f_y = interp1d(all_times, y_pix, kind='linear')


    '''
    Estimate the time stamp of each movie frame, aligned with the first eye frame timestamp

    The assumption is that the eye movie capture and film onset were synchronized. It's an approximation.
    Unfortunately, the movie frames have no time stamps, but the eye movie frames do

    The time between the movie's logged onset and offset is ~400s longer than the actual movie length
    (based on frame count and fps).

    The delay between the eyetracking/movie's logged onset (same trigger) and the first eye frame's
    timestamp is ~300ms... good enough? (as in, first movie frame is shown around the time
    when the first eye frame is captured?)

    '''

    return f_x, f_y


def drawgaze(clip,fx,fy,r_zone):
    """
    Adapted from: https://zulko.github.io/moviepy/examples/headblur.html
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


def headblur(clip,fx,fy,r_zone,r_blur=None):
    """
    From: https://zulko.github.io/moviepy/examples/headblur.html
    Returns a filter that will blurr a moving part (a head ?) of
    the frames. The position of the blur at time t is
    defined by (fx(t), fy(t)), the radius of the blurring
    by ``r_zone`` and the intensity of the blurring by ``r_blur``.
    Requires OpenCV for the circling and the blurring.
    Automatically deals with the case where part of the image goes
    offscreen.
    """

    if r_blur is None: r_blur = 2*r_zone/3

    def fl(gf,t):

        im = gf(t)
        h,w,d = im.shape
        x,y = int(fx(t)),int(fy(t))
        x1,x2 = max(0,x-r_zone),min(x+r_zone,w)
        y1,y2 = max(0,y-r_zone),min(y+r_zone,h)
        region_size = y2-y1,x2-x1

        mask = np.zeros(region_size).astype('uint8')
        cv2.circle(mask, (r_zone,r_zone), r_zone, 255, -1,
                   lineType=cv2.CV_AA)

        mask = np.dstack(3*[(1.0/255)*mask])

        orig = im[y1:y2, x1:x2]
        blurred = cv2.blur(orig,(r_blur, r_blur))
        im[y1:y2, x1:x2] = mask*blurred + (1-mask)*orig
        return im

    return clip.fl(fl)


def main():

    args = get_arguments()

    film_path = args.film
    fps = args.fps
    gaze_path = args.gaze

    start_frame = args.start_frame
    num_frames = args.num_frames

    if args.partial:
        start_time = start_frame / fps
        end_time = (start_frame + num_frames) / fps
        clip = VideoFileClip(film_path).subclip(start_time,end_time)
    else:
        (start_time, end_time) = (0.0, np.inf)
        clip = VideoFileClip(film_path)


    if args.deepgaze:
        dg_coords = pd.read_csv(gaze_path, sep = '\t').to_numpy()

        if args.partial:
            timeframes = np.arange(num_frames+1) / fps

            dg_x = dg_coords[start_frame:start_frame+num_frames+1, 1] # width
            dg_y = dg_coords[start_frame:start_frame+num_frames+1, 0] # height

        else:
            timeframes = np.arange(dg_coords.shape[0]) / fps

            dg_x = dg_coords[:, 1] # width
            dg_y = dg_coords[:, 0] # height


        f_xcoord = interp1d(timeframes, dg_x, kind='linear')
        f_ycoord = interp1d(timeframes, dg_y, kind='linear')

    else:
        f_xcoord, f_ycoord = get_eyetrack_function(gaze_path, conf_thresh=args.gaze_confthres, time_info=(start_time, end_time))

    clip_gaze = clip.fx( drawgaze, f_xcoord, f_ycoord, 10)

    if args.partial:
        clip_gaze.write_videofile(os.path.join(args.out_path, args.outname + str(int(start_time)) + '_' + str(int(end_time)) + '.mp4')
    else:
        clip_gaze.write_videofile(os.path.join(args.out_path, args.outname + '.mp4')


if __name__ == '__main__':
    sys.exit(main())
