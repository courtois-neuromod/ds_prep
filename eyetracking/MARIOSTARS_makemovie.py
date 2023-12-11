import csv
import json
import os, glob
import retro
retro.data.Integrations.add_custom_path('/home/labopb/Documents/Marie/neuromod/MarioStars/mariostars.stimuli')
#from retro.scripts.playback_movie import playback_movie

import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor as Executor

import numpy as np
import pandas as pd
from PIL import Image
from moviepy.editor import *
from scipy.interpolate import interp1d
import cv2

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

'''
General notes
On laptop, local conda env is 'movie_making'
Includes cv2, moviepy and gym-retro
see requirements_moviemaking.txt

Also needed: Mario's ROM, downloaded from https://github.com/courtois-neuromod/mario.stimuli
Doc on Installing ROMS : https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms
To install the game, as per instructions (RUN IT ONLY ONCE)
python3 -m retro.import /path/to/your/ROMs/directory/
e.g., python -m retro.import /home/labopb/Documents/Marie/neuromod/Mario/mario.stimuli
(mario.stimuli repo contains SuperMarioBros-Nes directory with file rom.nes in it)

Before running the script, the online gaze data files exported by pupil need
to be converted to .npz using the MARIOSTARS_convert_serialGazeAndFix_2_npz.py script
(On laptop, run in pupil_venv environment)
'''

def get_arguments():

    parser = argparse.ArgumentParser(description='Project gaze onto videogame replays and export movie file')
    parser.add_argument('--file_path', default='', type=str, help='absolute path to session directory w .bk2, log and gaze files')
    parser.add_argument('--gaze_path', default='', type=str, help='absolute path to directory w preprocessed gaze files (gaze.pldata -> .npz)')
    parser.add_argument('--out_path', default='', type=str, help='absolute path to output directory')
    parser.add_argument('--driftcorr', action='store_true', help='if true, gaze is corrected based on median gaze at fixations')
    parser.add_argument('--conf', default=0.98, type=float, help='confidence threshold to filter gaze data')
    parser.add_argument('--fixconf', default=0.90, type=float, help='confidence threshold to filter fixations data')
    args = parser.parse_args()

    return args


'''
# Functions below (playback_movie, load_movie, _play) from gym-retro (retro.playback_movie)
# Script from https://github.com/openai/retro/blob/master/retro/scripts/playback_movie.py
# TODO: modify to superimpose gaze directly onto reconstructed or played files (from .bk2), to reduce output / storage
'''
def playback_movie(emulator, movie, monitor_csv=None, video_file=None, info_file=None, npy_file=None, viewer=None, video_delay=0, lossless=None, record_audio=True):
    ffmpeg_proc = None
    viewer_proc = None
    info_steps = []
    actions = np.empty(shape=(0, emulator.num_buttons * movie.players), dtype=bool)
    if viewer or video_file:
        video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video.bind(('127.0.0.1', 0))
        vr = video.getsockname()[1]
        input_vformat = [
            '-r', str(emulator.em.get_screen_rate()),
            '-s', '%dx%d' % emulator.observation_space.shape[1::-1],
            '-pix_fmt', 'rgb24',
            '-f', 'rawvideo',
            '-probesize', '32',
            '-thread_queue_size', '10000',
            '-i', 'tcp://127.0.0.1:%i?listen' % vr
        ]
        if record_audio:
            audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            audio.bind(('127.0.0.1', 0))
            ar = audio.getsockname()[1]
            input_aformat = [
                '-ar', '%i' % emulator.em.get_audio_rate(),
                '-ac', '2',
                '-f', 's16le',
                '-probesize', '32',
                '-thread_queue_size', '60',
                '-i', 'tcp://127.0.0.1:%i?listen' % ar
            ]
        else:
            audio = None
            ar = None
            input_aformat = ['-an']
        stdout = None
        output = []
        if video_file:
            if not lossless:
                output = ['-c:a', 'aac', '-b:a', '128k', '-strict', '-2', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-f', 'mp4', '-pix_fmt', 'yuv420p', video_file]
            elif lossless == 'mp4':
                output = ['-c:a', 'aac', '-b:a', '192k', '-strict', '-2', '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '0', '-f', 'mp4', '-pix_fmt', 'yuv444p', video_file]
            elif lossless == 'mp4rgb':
                output = ['-c:a', 'aac', '-b:a', '192k', '-strict', '-2', '-c:v', 'libx264rgb', '-preset', 'veryslow', '-crf', '0', '-f', 'mp4', '-pix_fmt', 'rgb24', video_file]
            elif lossless == 'png':
                output = ['-c:a', 'flac', '-c:v', 'png', '-pix_fmt', 'rgb24', '-f', 'matroska', video_file]
            elif lossless == 'ffv1':
                output = ['-c:a', 'flac', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', '-f', 'matroska', video_file]
        if viewer:
            stdout = subprocess.PIPE
            output = ['-c', 'copy', '-f', 'nut', 'pipe:1']
        ffmpeg_proc = subprocess.Popen(['ffmpeg', '-y',
                                        *input_vformat,  # Input params (video)
                                        *input_aformat,  # Input params (audio)
                                        *output],  # Output params
                                       stdout=stdout)
        video.close()
        video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if audio:
            audio.close()
            audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        audio_connected = False

        time.sleep(0.3)
        try:
            video.connect(('127.0.0.1', vr))
        except ConnectionRefusedError:
            video.close()
            if audio:
                audio.close()
            ffmpeg_proc.terminate()
            raise
        if viewer:
            viewer_proc = subprocess.Popen([viewer, '-'], stdin=ffmpeg_proc.stdout)
    frames = 0
    score = [0] * movie.players
    reward_fields = ['r'] if movie.players == 1 else ['r%d' % i for i in range(movie.players)]
    wasDone = False

    def killprocs(*args, **kwargs):
        ffmpeg_proc.terminate()
        if viewer:
            viewer_proc.terminate()
            viewer_proc.wait()
        raise BrokenPipeError

    def waitprocs():
        if ffmpeg_proc:
            video.close()
            if audio:
                audio.close()
            if not viewer_proc or viewer_proc.poll() is None:
                ffmpeg_proc.wait()

    while True:
        if movie.step():
            keys = []
            for p in range(movie.players):
                for i in range(emulator.num_buttons):
                    keys.append(movie.get_key(i, p))
            if npy_file:
                actions = np.vstack((actions, (keys,)))
        elif video_delay < 0 and frames < -video_delay:
            keys = [0] * emulator.num_buttons
        else:
            break
        display, reward, done, info = emulator.step(keys)
        if info_file:
            info_steps.append(info)
        if movie.players > 1:
            for p in range(movie.players):
                score[p] += reward[p]
        else:
            score[0] += reward
        frames += 1
        try:
            if hasattr(signal, 'SIGCHLD'):
                signal.signal(signal.SIGCHLD, killprocs)
            if viewer_proc and viewer_proc.poll() is not None:
                break
            if ffmpeg_proc and frames > video_delay:
                video.sendall(bytes(display))
                if audio:
                    sound = emulator.em.get_audio()
                    if not audio_connected:
                        time.sleep(0.2)
                        audio.connect(('127.0.0.1', ar))
                        audio_connected = True
                    if len(sound):
                        audio.sendall(bytes(sound))
        except BrokenPipeError:
            waitprocs()
            raise
        finally:
            if hasattr(signal, 'SIGCHLD'):
                signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        if done and not wasDone:
            if monitor_csv:
                monitor_csv.writerow({**dict(zip(reward_fields, score)), 'l': frames, 't': frames / 60.0})
            frames = 0
            score = [0] * movie.players
        wasDone = done
    if hasattr(signal, 'SIGCHLD'):
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    if monitor_csv and frames:
        monitor_csv.writerow({**dict(zip(reward_fields, score)), 'l': frames, 't': frames / 60.0})
    if npy_file:
        kwargs = {
            'actions': actions
        }
        if info_file:
            kwargs['info'] = info_steps
        try:
            np.savez_compressed(npy_file, **kwargs)
        except IOError:
            pass
    elif info_file:
        try:
            with open(info_file, 'w') as f:
                json.dump(info_steps, f)
        except IOError:
            pass
    waitprocs()


def load_movie(movie_file):
    movie = retro.Movie(movie_file)
    duration = -1
    while movie.step():
        duration += 1
    movie = retro.Movie(movie_file)
    #movie.step()
    emulator = retro.make(game=movie.get_game(),
                          state=retro.State.NONE,
                          use_restricted_actions=retro.Actions.ALL,
                          players=movie.players,
                          inttype=retro.data.Integrations.CUSTOM_ONLY)
    data = movie.get_state()
    emulator.initial_state = data
    emulator.reset()
    return emulator, movie, duration

'''
def _play(movie, args, monitor_csv):
    video_file = None
    info_file = None
    npy_file = None
    if args.lossless in ('png', 'ffv1'):
        ext = '.mkv'
    else:
        ext = '.mp4'

    basename = os.path.splitext(movie)[0]
    if not args.no_video:
        video_file = basename + ext
    if args.info_dict:
        info_file = basename + '.json'
    if args.npy_actions:
        npy_file = basename + '.npz'
    while True:
        emulator = None
        try:
            emulator, m, duration = load_movie(movie)
            if args.ending is not None:
                if args.ending < 0:
                    delay = duration + args.ending
                else:
                    delay = -(duration + args.ending)
            else:
                delay = 0
            playback_movie(emulator, m, monitor_csv, video_file, info_file, npy_file, args.viewer, delay, args.lossless, not args.no_audio)
            break
        except ConnectionRefusedError:
            pass
        except RuntimeError:
            if not os.path.exists(movie):
                raise FileNotFoundError(movie)
            raise
        finally:
            del emulator
'''


'''
Support functions to convert gaze from normalized to pixel space, superimpose onto movie files and export into .mp4

Notes on scaling and timing:

Frames per second (fps) for Friends is 29.97
Frames per second (fps) for Mario is 60

Full screen is 1280:1024
Mario, like the screen, is 4:5 and supposedly projected onto the entire screen (100% of normalized height and width)
No padding adjustments needed

Mario frames are (224, 240) height = 224 pixels, width = 240 pixels (file dims)
Frames are stretched into 1280:1024 (on screen)

By comparison, Friends is projected onto 4:3 using the screen's entire width,
It is centered along the height dimension w padding above and below -> 1280:960
Friends frame sizes = 720:480 pixels (file dims)
Frames are projected onto the full screen (with stretching) with some padding along height (no padding along width)
The padded height corresponds to 512 pixels
480 -> 512 pixels (16 pixels above and below on height axis)
Padded frames are then stretched into 1280:960 (on screen)


Friends: Estimate the time stamp of each movie frame, aligned with the first eye frame timestamp

The assumption is that the eye movie capture and film onset were synchronized. It's an approximation.
Unfortunately, the movie frames have no time stamps, but the eye movie frames do

The time between the movie's logged onset and offset is ~400s longer than the actual movie length
(based on frame count and fps).

The delay between the eyetracking/movie's logged onset (same trigger) and the first eye frame's
timestamp is ~300ms... good enough? (as in, first movie frame is shown around the time
when the first eye frame is captured?)


Mario: precise .bk2 onset times logged into event and log files ; timed with same clock as eyetracking frames
Realignment is precise without approximations

TODO: check that logged times for different steps correspond to target interval for the .bk2s

'''

def norm2pix(x_norms, y_norms):
    '''
    Function converts gaze from normalized space (0-100% of screen) to pixel space
    Cartesian coordinates x and y as percent of screen -> coordinates h, w on image frame (array))
    x_norms (1D array of float): normalized gaze position coordinate along WIDTH
    y_norms (1D array of float):  normalized gaze position coordinate along HEIGHT
    '''
    assert len(x_norms)==len(y_norms)

    screen_size = (1280, 1024)
    #img_shape = (720, 480) # Friends
    img_shape = (240, 224) # Mario

    '''
    # Friends
    # x (width) is projected along full width of the screen, so 100% = 720
    x_pix = np.floor(x_norms*720).astype(int)
    x_pix[x_pix > 719] = 719
    x_pix[x_pix < 0] = 0

    # y (height) is projected along 960/1024 of the screen height, so 93.75% = 480 and 100% = 512 (480*1024/960)
    # also, y coordinate is flipped to project onto stimulus (movie frame)
    y_pix = np.floor((1 - y_norms)*512).astype(int)
    y_pix -= 16 # remove padding to realign gaze onto unpadded frames

    y_pix[y_pix > 479] = 479
    y_pix[y_pix < 0] = 0
    '''

    # Mario
    # x (width) is projected along full width of the screen, so 100% = 240
    x_pix = np.floor(x_norms*240).astype(int)
    x_pix[x_pix > 239] = 239
    x_pix[x_pix < 0] = 0

    # y (height) is projected along 100% of the screen height, so 100% = 224
    # also, y coordinate is flipped to project onto stimulus (movie frame)
    y_pix = np.floor((1 - y_norms)*224).astype(int)
    y_pix[y_pix > 223] = 223
    y_pix[y_pix < 0] = 0

    #return list(zip(y_pix.tolist(), x_pix.tolist()))
    return x_pix.tolist(), y_pix.tolist()


def get_eyetrack_function(gaze_file, time_0 = 0, d3=False, conf_thresh=0.98, fix_time = None, fixconf=0.9):
    '''
    Function formats subject's eyetracking data (from normalized gaze to pixel space),
    and outputs functions that extrapolate gaze position (in x and y) as a function of onset time

    If fix_time is not None, then the gaze is corrected based on the 5s fixation that precedes the onset of the level

    gaze_file (str): '/path/to/2d_gaze.npz'
    e.g., '/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-01/ses-039/2d_gaze.npz'

    time_0 (float) : onset time of the .bk2 file of interest

    d3 (bool): gaze is from 3d model
    conf_thresh (float): confidence threshold for gaze estimation
    '''

    run_gaze = np.load(gaze_file, allow_pickle=True)
    if d3:
        gz = run_gaze['gaze3d']
    else:
        gz = run_gaze['gaze2d']

    # Assume no drift
    fix_position = [0.5, 0.5]

    if fix_time is not None:
    #if fix_file is not None:
        #run_fix = np.load(fix_file, allow_pickle=True)['fixations']

        # time after fixation onset for gaze to stabilize
        buffer_time = 0.8
        end_fix = 4.0

        fix_coord = []

        #for fix in run_fix:
        for gaze in gz:
            if gaze['confidence'] > fixconf:
                if gaze['timestamp'] > (fix_time[0] + buffer_time) and gaze['timestamp'] < (fix_time[0] + end_fix):
                    fix_coord.append(gaze['norm_pos'])

        if len(fix_coord) > 0:
            # todo : update fix_position
            coord_arr = np.array(fix_coord)
            x_arr, y_arr = coord_arr[:, 0], coord_arr[:, 1]
            fix_position = [np.median(x_arr), np.median(y_arr)]

    # Export normalized x and y coordinates, and their corresponding time stamp

    # Friends: (time is relative to the first eye frame capture = "time 0") for each gaze

    # Mario: .bk2 file onset time is saved (same clock as eyetracking frames),
    # .bk2 onset time must be entered as "time_0" to align gaze coordinates with .bk2 onset

    # build array of x, y and timestamps
    all_x = []
    all_y = []
    all_times = []

    print(len(gz))
    for i in range(len(gz)):
        gaze = gz[i]
        if gaze['confidence'] > conf_thresh:
            x_norm, y_norm = gaze['norm_pos']
            x_norm = x_norm - (fix_position[0] - 0.5)
            y_norm = y_norm - (fix_position[1] - 0.5)
            timestp = gaze['timestamp'] - time_0
            #timestp = gaze['timestamp'] # - gz[0]['timestamp']
            all_x.append(x_norm)
            all_y.append(y_norm)
            all_times.append(timestp)

    '''
    If low (below threshold) confidence gaze at edges, the function won't model and extrapolate to entire distribution
    Probably not needed for Mario unless lots of missing eye frames, to be determined and flagged during eyetracking QC
    if all_times[0] > 0:
        all_times = [0.0] + all_times
        x, y = gz[2]['norm_pos']
        all_x = [x] + all_x
        all_y = [y] + all_y

    if all_times[-1] < gz[-1]['timestamp'] - gz[0]['timestamp']:
        all_times.append(gz[-1]['timestamp'] - gz[0]['timestamp'])
        x, y = gz[-1]['norm_pos']
        all_x.append(x_norm)
        all_y.append(y_norm)
    '''

    print(str(100*(len(all_times)/len(gz))) + '% of gaze above confidence threshold')

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

    return f_x, f_y


def drawgaze(clip,fx,fy,r_zone):
    """
    I modified this function from headblur to draw a red circle on an image instead of blurring a circle in the image:
    https://zulko.github.io/moviepy/examples/headblur.html
    https://github.com/Zulko/moviepy/blob/master/moviepy/video/fx/headblur.py

    https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
    https://www.geeksforgeeks.org/python-opencv-cv2-blur-method/

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
        circled = cv2.circle(orig, (r_zone, r_zone), r_zone, (100, 100, 155), -1,
                             lineType=cv2.CV_AA)

        im[y1:y2, x1:x2] = circled
        return im

    return clip.fl(fl)


def create_timingdict(dir_path):
    timing_data = {}
    timing_data['dir_path'] = dir_path

    #'/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/pupil_data/sub-01/ses-002/
    #event_list = sorted(glob.glob(dir_path + '/*task-mario_run-*_events.tsv'))
    event_list = sorted(glob.glob(dir_path + '/*task-mariostars_run-*_events.tsv'))

    print(event_list)

    for ev_path in event_list:
        run_num = ev_path[-12]
        file_num = os.path.basename(ev_path).split('_')[2]
        print(file_num, run_num)

        if not file_num in timing_data:
            timing_data[file_num] = {}

        timing_data[file_num]['run_' + run_num] = {}
        timing_data[file_num]['run_' + run_num]['bk2_dict'] = {}

        ev_file = pd.read_csv(ev_path, sep = '\t')

        for i in range(ev_file.shape[0]):
            if ev_file['trial_type'][i] == 'gym-retro_game':
                tstamp = ev_file['sample'][i]
                fname = os.path.basename(ev_file['stim_file'][i])
                fix_onset = ev_file['sample'][i-1]
                fix_time = [fix_onset, fix_onset]

                timing_data[file_num]['run_' + run_num]['bk2_dict'][fname] = {}
                timing_data[file_num]['run_' + run_num]['bk2_dict'][fname]['start_event'] = tstamp
                timing_data[file_num]['run_' + run_num]['bk2_dict'][fname]['fixation_time'] = fix_time

    print(timing_data)
    return timing_data


def export_gazemovies(timing_dict, gaze_path, out_path, conf_threshold, driftcorr=False, fixconf=0.9):

    ext = '.mp4' #'.mkv' #'.mp4'
    delay = 0
    monitor_csv = None
    info_file = None
    npy_file = None
    viewer = None
    lossless = 'png' #'mp4'
    reccord_audio = True #False

    dir_path = timing_dict['dir_path']

    for data_num in timing_dict.keys():
        if data_num != 'dir_path':
            for run in timing_dict[data_num].keys():
                rnum = run[-1]
                run_dict = timing_dict[data_num][run]['bk2_dict']

                run_gpath = gaze_path + '/' + data_num + '_run-0' + rnum + '_online_gaze2D.npz'
                #run_fpath = gaze_path + '/' + data_num + '_run-0' + rnum + '_online_fixations.npz' if driftcorr else None

                for bk2 in run_dict.keys():
                    onset_time = run_dict[bk2]['start_event']
                    fixation_time = run_dict[bk2]['fixation_time'] if driftcorr else None

                    try:
                        emulator.close()
                    except:
                        print('no emulator')

                    bk2_path = os.path.join(dir_path, bk2)
                    video_file = os.path.join(out_path, bk2.split('.')[0] + ext)

                    '''
                    try:
                        emulator, m, duration = load_movie(bk2_path)
                        playback_movie(emulator, m, monitor_csv, video_file, info_file, npy_file, viewer, delay, lossless, reccord_audio)
                        emulator.close()

                        #f_xcoord, f_ycoord = get_eyetrack_function(run_gpath, time_0 = onset_time, conf_thresh=conf_threshold, fix_file=run_fpath, fix_time=fixation_time, fixconf=fixconf)
                        f_xcoord, f_ycoord = get_eyetrack_function(run_gpath, time_0 = onset_time, conf_thresh=conf_threshold, fix_time=fixation_time, fixconf=fixconf)
                        clip = VideoFileClip(video_file)

                        clip_gaze = clip.fx(drawgaze, f_xcoord, f_ycoord, 5)

                        file_ext = '_wgazeDC.mp4' if driftcorr else '_wgaze.mp4'
                        clip_gaze.write_videofile(video_file.split('.')[0] + file_ext)

                    except:
                        print('Something went wrong processing file ' + bk2)
                    '''

                    emulator, m, duration = load_movie(bk2_path)
                    playback_movie(emulator, m, monitor_csv, video_file, info_file, npy_file, viewer, delay, lossless, reccord_audio)
                    emulator.close()

                    #f_xcoord, f_ycoord = get_eyetrack_function(run_gpath, time_0 = onset_time, conf_thresh=conf_threshold, fix_file=run_fpath, fix_time=fixation_time, fixconf=fixconf)
                    f_xcoord, f_ycoord = get_eyetrack_function(run_gpath, time_0 = onset_time, conf_thresh=conf_threshold, fix_time=fixation_time, fixconf=fixconf)
                    clip = VideoFileClip(video_file)

                    clip_gaze = clip.fx(drawgaze, f_xcoord, f_ycoord, 5)

                    file_ext = '_wgazeDC.mp4' if driftcorr else '_wgaze.mp4'
                    clip_gaze.write_videofile(video_file.split('.')[0] + f'_run-0{rnum}' + file_ext)


if __name__ == '__main__':
    args = get_arguments()

    # create dictionary of onset times per .bk2
    file_path = args.file_path
    # e.g., '/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/pupil_data/sub-01/ses-002'
    timing_dict = create_timingdict(file_path)

    conf_threshold = args.conf
    gaze_path = args.gaze_path
    # e.g., /home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/offline_calibration/sub-01/ses-002
    out_path = args.out_path
    # e.g., '/home/labopb/Documents/Marie/neuromod/Mario/tests'
    driftcorr = args.driftcorr
    fixconf = args.fixconf

    export_gazemovies(timing_dict, gaze_path, out_path, conf_threshold, driftcorr, fixconf)
