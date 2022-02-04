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
parser.add_argument('--code_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--data_dir', default='', type=str, help='absolute path to main data directory')
parser.add_argument('--sub', default='config.json', type=str, help='subject number')
parser.add_argument('--ses', default='config.json', type=str, help='session number')
args = parser.parse_args()

sys.path.append(os.path.join(args.code_dir, "pupil", "pupil_src", "shared_modules"))
#sys.path.append(os.path.join("/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking", "pupil", "pupil_src", "shared_modules"))
from video_capture.file_backend import File_Source
from file_methods import PLData_Writer, load_pldata_file, load_object, save_object
from gaze_producer.worker.fake_gpool import FakeGPool, FakeIPC

from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
from gaze_mapping.gazer_2d import Gazer2D
from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
from gaze_mapping.gazer_3d.gazer_headset import Gazer3D


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


def curate_list(template_json, ordered_file_list, ordered_file_type):
    '''
    For each run, file order should be: calin pupils, calib params, run pupils
    '''
    runs = []
    for idx in range(len(ordered_file_type)-2):
        if ordered_file_type[idx] == 'CALIB PUPILS':
            if ordered_file_type[idx+1] == 'CALIB PARAM':
                if ordered_file_type[idx+2] == 'RUN PUPILS':

                    run_num = ordered_file_list[idx+2][-18]
                    runs.append(run_num)
                    print('Run ' + run_num)
                    # calibration eye movie and online data (gaze and pupils)
                    template_json['run' + run_num + '_calib_mp4'] = ordered_file_list[idx][:-13] + '/eye0.mp4'
                    #template_json['run' + run_num + '_calib_mp4'] = ordered_file_list[idx]
                    template_json['run' + run_num + '_run_mp4'] = ordered_file_list[idx+2][:-13] + '/eye0.mp4'
                    template_json['run' + run_num + '_calibration_data'] = ordered_file_list[idx+1]

                    template_json['run' + run_num + '_calibration_parameters_2d'] = "/placeholder.plcal"
                    template_json['run' + run_num + '_calibration_parameters_3d'] = "/placeholder.plcal"

    template_json['runs'] = runs

    return template_json


def update_json(template_json, data_path):
    '''
    Script determines chronological order in which files were acquired

    FILES to order:
    online calibration pupils exported by software
    online calibration pupils saved in the calibration parameters file (Basile's)
    online main run pupils exported by software

    '''
    list_firstTstamp = []
    list_filename = []
    list_filetype = []
    list_len = []

    # Obtain time stamp of first pupil of calibration param files in directory
    list_calibparam_files = glob.glob(data_path + '/*npz')
    for cp_file in list_calibparam_files:
        try:
            pupils = np.load(cp_file, allow_pickle=True)['pupils']
            list_firstTstamp.append(pupils[0]['timestamp'])
            list_filename.append(os.path.basename(cp_file))
            list_filetype.append('CALIB PARAM')
            list_len.append(len(pupils))
        except:
            print('File ' + cp_file + ' did not load.')


    # Obtain time stamp of first pupil of calibration pupils (detected online) in directory

    list_calib_pupils = glob.glob(data_path + '/*pupil/*yeTracker*alibration*/*/pupil.pldata')
    for cpupils in list_calib_pupils:
        chunks = cpupils.split('/')[-4:]
        chunks = os.path.join(chunks[-4], chunks[-3], chunks[-2], chunks[-1])
        try:
            pupils = load_pldata_file(cpupils[:-13], 'pupil')[0]
            list_firstTstamp.append(pupils[0]['timestamp'])
            list_filename.append(chunks)
            list_filetype.append('CALIB PUPILS')
            list_len.append(len(pupils))
        except:
            print('File ' + chunks + ' did not load.')
    '''
    list_calib_pupils = glob.glob(data_path + '/*pupil/EyeTracker-Calibration*/*/eye0.mp4')
    for cpupils in list_calib_pupils:
        chunks = cpupils.split('/')[-4:]
        chunks = os.path.join(chunks[-4], chunks[-3], chunks[-2], chunks[-1])
        try:
            g_pool = make_detection_gpool()
            calib_eye_file = File_Source(g_pool, source_path=cpupils).timestamps
            #pupils = load_pldata_file(cpupils[:-13], 'pupil')[0]
            list_firstTstamp.append(calib_eye_file[0])
            list_filename.append(chunks)
            list_filetype.append('CALIB PUPILS')
            list_len.append(len(calib_eye_file))
        except:
            print('File ' + chunks + ' did not load.')
    '''
    # Obtain time stamp of first pupil of main run pupils (detected online) in directory
    list_run_pupils = glob.glob(data_path + '/*pupil/task-thingsmemory*/*/pupil.pldata')
    for rpupils in list_run_pupils:
        chunks = rpupils.split('/')[-4:]
        chunks = os.path.join(chunks[-4], chunks[-3], chunks[-2], chunks[-1])
        try:
            pupils = load_pldata_file(rpupils[:-13], 'pupil')[0]
            list_firstTstamp.append(pupils[0]['timestamp'])
            list_filename.append(chunks)
            list_filetype.append('RUN PUPILS')
            list_len.append(len(pupils))
        except:
            print('File ' + chunks + ' did not load.')

    # TODO: indices of sorted timestamps (each file's first timestamp)
    sort_index = np.argsort(np.array(list_firstTstamp))
    ordered_file_list = []
    ordered_file_type = []

    for idx in sort_index:
        print(list_firstTstamp[idx], list_len[idx], list_filetype[idx], list_filename[idx])
        ordered_file_list.append(os.path.join(data_path, list_filename[idx]))
        ordered_file_type.append(list_filetype[idx])

    session_json = curate_list(template_json, ordered_file_list, ordered_file_type)

    return session_json


if __name__ == "__main__":
    '''
    Script generates config file for a subject's THINGS session

    The config file is compatible with the offline calibration script (offline_calibration_THINGS.py)
    and the two QC scripts (summary: quality_check_THINGS_summary.py, and more extensive: quality_check_THINGS.py)

    '''
    # Load template file

    template_path = os.path.join(args.code_dir, 'config', 'config_THINGS', 'config_template.json')

    with open(template_path, 'r') as f:
        template_json = json.load(f)

    sub_num = args.sub
    ses_num = args.ses

    template_json['subject'] = sub_num
    template_json['session'] = ses_num[-3:]
    template_json['out_dir'] = os.path.join(args.data_dir, 'offline_calibration', 's' + sub_num[-2:], ses_num)

    session_json = update_json(template_json, os.path.join(args.data_dir, 'pupil_data', sub_num, ses_num))


    out_path = os.path.join(args.code_dir, 'config', 'config_THINGS', 'config_THINGS_s' + sub_num[-2:] + '_ses' + ses_num[-3:] + '.json')

    with open(out_path, 'w') as outfile:
        json.dump(session_json, outfile)
