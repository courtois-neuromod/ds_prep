import os, sys, platform, json, glob
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
parser.add_argument('--sub', default='config.json', type=str, help='absolute path to config file')
parser.add_argument('--ses', default='config.json', type=str, help='absolute path to config file')
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


'''
Quality checks: contrast two sets of pupils and gaze

1. (Optional) Flag missing frames in eye movie (mp4) based on gaps in camera timestamps
2. Flag percentage of pupils and gaze under confidence threshold
3. Flag percentage of gaze outside screen area
4. Plot x and z pupil and gaze position over time
'''

if __name__ == "__main__":
    '''
    Script determines chronological order in which files were acquired

    FILES to order:
    calib pupils / gaze: which file...
    calib parameters : which correspond to calib data
    run pupils / gaze (should be same)

    '''

    list_firstTstamp = []
    list_filename = []
    list_len = []

    data_path = '/data/neuromod/DATA/fmri_tmp/things/sourcedata/' + args.sub + '/'+ args.ses

    # Obtain time stamp of first pupil of calibration param files in directory
    list_calibparam_files = glob.glob(data_path + '/*npz')
    for cp_file in list_calibparam_files:
        try:
            pupils = np.load(cp_file, allow_pickle=True)['pupils']
            list_firstTstamp.append(pupils[0]['timestamp'])
            list_filename.append('CALIB PARAM : ' + os.path.basename(cp_file))
            list_len.append(len(pupils))
        except:
            print('File ' + cp_file + ' did not load.')


    # Obtain time stamp of first pupil of calibration pupils in directory
    list_calib_pupils = glob.glob(data_path + '/*pupil/EyeTracker-Calibration/*/pupil.pldata')
    for cpupils in list_calib_pupils:
        chunks = cpupils.split('/')[-4:]
        chunks = os.path.join(chunks[-4], chunks[-3], chunks[-2], chunks[-1])
        try:
            pupils = load_pldata_file(cpupils[:-13], 'pupil')[0]
            list_firstTstamp.append(pupils[0]['timestamp'])
            list_filename.append('CALIB PUPILS :' + chunks)
            list_len.append(len(pupils))
        except:
            print('File ' + chunks + ' did not load.')


    # Obtain time stamp of first pupil of main run pupils in directory
    list_run_pupils = glob.glob(data_path + '/*pupil/task-thingsmemory*/*/pupil.pldata')
    for rpupils in list_run_pupils:
        chunks = rpupils.split('/')[-4:]
        chunks = os.path.join(chunks[-4], chunks[-3], chunks[-2], chunks[-1])
        try:
            pupils = load_pldata_file(rpupils[:-13], 'pupil')[0]
            list_firstTstamp.append(pupils[0]['timestamp'])
            list_filename.append('RUN PUPILS :' + chunks)
            list_len.append(len(pupils))
        except:
            print('File ' + chunks + ' did not load.')

    # TODO: indices of sorted timestamps (each file's first timestamp)
    sort_index = np.argsort(np.array(list_firstTstamp))

    for idx in sort_index:
        print(list_firstTstamp[idx], list_len[idx], list_filename[idx])
