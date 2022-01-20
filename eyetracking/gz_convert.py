import os, sys, platform, json
import numpy as np
from types import SimpleNamespace

import math
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Perform off-line gaze mapping with 2D and 3D pupil detectors ')
parser.add_argument('--infile', default='', type=str, help='absolute path to main input file')
parser.add_argument('--run_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--outfile', default='config.json', type=str, help='absolute path to config file')
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


if __name__ == "__main__":
    '''
    ...
    outfile = 'path/to/run_s2e04a_gaze2D.npz'
    '''
    seri_gaze = load_pldata_file(args.infile, 'gaze')[0]

    print(len(seri_gaze))

    deserialized_gaze = []

    # Convert serialized file to list of dictionaries...
    for gaze in seri_gaze:
        gaze_data = {}
        for key in gaze.keys():
            if key != 'base_data':
                gaze_data[key] = gaze[key]
        deserialized_gaze.append(gaze_data)

    print(len(deserialized_gaze))

    np.savez(args.outfile, gaze2d = deserialized_gaze)
