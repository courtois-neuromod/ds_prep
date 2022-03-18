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
    Script takes serialized gaze file and converts it to .npz (same format as offline gaze) so it is compatible with QC scripts
    The goal is to contrast online and offline gaze w QC scripts

    To call script for Friends dataset in console

    RUNDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"

    s1 ses39
    IFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/pupil_data/sub-01/ses-039/sub-01_ses-039_20211020-180629.pupil/task-friends-s2e4a/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-01/ses-039/run_s2e04a_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"

    s1 ses40
    IFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/pupil_data/sub-01/ses-040/sub-01_ses-040_20211103-181444.pupil/task-friends-s2e4b/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-01/ses-040/run_s2e04b_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"

    s2 ses43
    IFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/pupil_data/sub-02/ses-043/sub-02_ses-043_20211020-114642.pupil/task-friends-s2e4a/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-02/ses-043/run_s2e04a_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"

    s3 ses68
    IFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/pupil_data/sub-03/ses-068/sub-03_ses-068_20211013-140928.pupil/task-friends-s2e4a/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-03/ses-068_offline/run_s2e04a_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"

    s3 ses69
    IFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/pupil_data/sub-03/ses-069/sub-03_ses-069_20211027-141144.pupil/task-friends-s2e4b/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-03/ses-069/run_s2e04b_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"

    s6 ses40
    IFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/pupil_data/sub-06/ses-040/sub-06_ses-040_20211020-155839.pupil/task-friends-s2e4a/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-06/ses-040/run_s2e04a_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"

    s6 ses41
    IFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/pupil_data/sub-06/ses-041/sub-06_ses-041_20211027-160734.pupil/task-friends-s2e4b/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-06/ses-041/run_s2e04b_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"

    # Mario
    sub-01 ses-002 runs 1-3
    RUNDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
    IFILE="/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/pupil_data/sub-01/ses-002/sub-01_ses-002_20210611-105659.pupil/task-mario_run-03/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/offline_calibration/sub-01/ses-002/run3_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"

    sub-03 ses-003 runs 1-4
    RUNDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
    IFILE="/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/pupil_data/sub-03/ses-003/sub-03_ses-003_20211104-173758.pupil/task-mario_run-01/000"
    OFILE="/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/offline_calibration/sub-03/ses-003/run1_online_gaze2D.npz"
    python -m convert_serialGaze_2_npz --infile="${IFILE}" --run_dir="${RUNDIR}" --outfile="${OFILE}"


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
