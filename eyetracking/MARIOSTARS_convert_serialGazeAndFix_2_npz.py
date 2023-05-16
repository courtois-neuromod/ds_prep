import os, sys, platform, json, glob
import numpy as np
from types import SimpleNamespace

import math
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from pathlib import Path

import argparse


def get_arguments():

    parser = argparse.ArgumentParser(description='Deserialize online gaze and fixation data (pupil) from .pldata to .npz format')
    parser.add_argument('--in_path', default='', type=str, help='absolute path to input directory')
    parser.add_argument('--code_dir', default='/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking', type=str, help='absolute path to main code directory')
    parser.add_argument('--fixations', action='store_true', help='if true, script also serializes fixation data')
    parser.add_argument('--out_path', default='config.json', type=str, help='absolute path to output directory')
    args = parser.parse_args()

    return args


def serialize_pupil_data(in_path, out_path, fix):

    gaze_file_list = sorted(glob.glob(os.path.join(in_path, '*.pupil', 'task-mariostars*/*', 'gaze.pldata')))
    #gaze_file_list = sorted(glob.glob(os.path.join(in_path, '*.pupil', 'eyeTrackercalib-validate*/*', 'gaze.pldata')))

    for gaze_file in gaze_file_list:

        seri_gaze = load_pldata_file(gaze_file[:-12], 'gaze')[0]
        if fix:
            seri_fix = load_pldata_file(gaze_file[:-12], 'fixations')[0]
            print(len(seri_gaze), len(seri_fix))
        else:
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

        if fix:
            deserialized_fix = []

            for fix in seri_fix:
                fix_data = {}
                for key in fix.keys():
                    if key != 'base_data':
                        fix_data[key] = fix[key]
                deserialized_fix.append(fix_data)

            print(len(deserialized_fix))

        #d = gaze_file.split('/')[-4].split('_')[-1].split('.')[0]
        #r = gaze_file.split('/')[-3].split('-')[-1]
        #gaze_name = os.path.join(out_path, d + '_' + r + '_online_gaze2D.npz')
        #fix_name = os.path.join(out_path, d + '_' + r + '_online_fixations.npz')
        d_num = gaze_file.split('/')[-4].split('_')[-1].split('.')[0]
        r_num = gaze_file.split('/')[-3].split('_')[-1]
        gaze_name = os.path.join(out_path, d_num + '_' + r_num + '_online_gaze2D.npz')
        np.savez(gaze_name, gaze2d = deserialized_gaze)

        if fix:
            fix_name = os.path.join(out_path, d_num + '_' + r_num + '_online_fixations.npz')
            np.savez(fix_name, fixations = deserialized_fix)


if __name__ == '__main__':
    args = get_arguments()

    sys.path.append(os.path.join(args.code_dir, "pupil", "pupil_src", "shared_modules"))
    #sys.path.append(os.path.join("/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking", "pupil", "pupil_src", "shared_modules"))
    from video_capture.file_backend import File_Source
    from file_methods import PLData_Writer, load_pldata_file, load_object, save_object
    from gaze_producer.worker.fake_gpool import FakeGPool, FakeIPC

    from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
    from gaze_mapping.gazer_2d import Gazer2D
    from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
    from gaze_mapping.gazer_3d.gazer_headset import Gazer3D

    in_path = args.in_path
    # e.g., '/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/pupil_data/sub-01/ses-001'
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    # e.g., /home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/offline_calibration/sub-01/ses-001
    fix = args.fixations

    serialize_pupil_data(in_path, out_path, fix)
