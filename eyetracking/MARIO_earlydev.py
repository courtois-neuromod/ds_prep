import os, sys, platform, json
import numpy as np
import pandas as pd
from types import SimpleNamespace

import math
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Exports summary of QC measures for a single run of THINGS eyetracking data')
parser.add_argument('--code_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--config', default='config.json', type=str, help='absolute path to config file')
#parser.add_argument('--driftcor', action='store_true', help='if true, apply crude drift correction to gaze over run')
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

# Add Gym-retro to pupil_venv
from retro.scripts import playback_movie

        '''
        min_ratio = min(
            exp_win.size[0] / self._first_frame.shape[1],
            exp_win.size[1] / self._first_frame.shape[0],
        )
        width = int(min_ratio * self._first_frame.shape[1])
        height = int(min_ratio * self._first_frame.shape[0])

        self.game_vis_stim = visual.ImageStim(
            exp_win,
            size=(width, height),
            units="pixels",
            interpolate=False,
            flipVert=True,
            autoLog=False,
        )

        self.game_fps  # From Basile: game fps is 60

        Screen size: size=array([1280, 1024]

        Calibration routine
        Movie: ori=0.0, pos=array([0., 0.]), size=array([922., 720.]
        fps=UNKNOWN
        unnamed MovieStim2: size = (1280.0, 999.5661605206075)

        MARIO: utilise le plus d'ecran possible, et ratio est 4:5 comme l'ecran, donc full screen? (likely)
        Screen size: size=array([1280, 1024]
        array size TBD: check output (frames) from FP and Yan's code


        TODO: align .bk2 time stamps with eye frame time stamps (are they in the same units?), produce gaze segments that align w .bk2s (several per run)

        e.g., in log file: VideoGame: recording movie in /scratch/neuromod/data/mario/sourcedata/sub-01/ses-002/sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level3-2_004.bk2


        # In log files,
        name .bk2 file, GO: eyetracking and level; markers: level step... 0-4500
        Time logs same for pupils and in log file? different convention...


        LINKS:
        code de Yan: https://github.com/courtois-neuromod/shinobi_behav/blob/master/shinobi_behav/misc/check_deterministic.py

        code de Francois: https://github.com/courtois-neuromod/video_transformer/blob/main/src/datasets/presses_to_frames.py

        code de Gym Retro:
        En checkant la doc de gym retro, j’ai vu qu’ils ont une méthode built-in pour générer des vidéos à partir des bk2 :

        python3 -m retro.scripts.playback_movie Airstriker-Genesis-Level1-000000.bk2

        code: https://github.com/openai/retro/blob/master/retro/scripts/playback_movie.py


        de Basile: projeter le gaze direct dans le bk2 en le modifiant

        # convert time stamp from eyeframe to session log??

        Start e-track recc: 1623423720.6354
        Stop e-track recc: 1623424378.4611
        '''

# TODO: open bk2, convert to frames? mp4?
# Next level: add gaze directly in bk2

# split run's gaze files based on timing (log files)
# gaze that correspond to .bk2 filess

# 3 runs... one log file
# run 1
#1623423720.6354 	EXP 	fMRI TTL 0
#1623423720.6354 	INFO 	GO
#1623423720.6354 	INFO 	starting eyetracking recording

# name.bk2, level step 0, level step last?
#1623423690.0039 ?
run_1 = [('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_000.bk2', 1623423720.6492, 1623423796.618612),
         ('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_001.bk2', 1623423796.6371, 1623423859.648054),
         ('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_002.bk2', 1623423859.6651, 1623423946.362529),
         ('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_003.bk2', 1623423946.3814, 1623424014.461255),
         ('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_004.bk2', 1623424014.4783, 1623424065.296016),
         ('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_005.bk2', 1623424065.3126, 1623424130.344425),
         ('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_006.bk2', 1623424130.3622, 1623424218.756507),
         ('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_007.bk2', 1623424218.7748, 1623424271.508041),
         ('sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_008.bk2', 1623424271.5262, 1623424348.170344)]

online = True
sgaze_path = '/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/pupil_data/sub-01/ses-002/sub-01_ses-002_20210611-105659.pupil/task-mario_run-01/000'
gaze_path = '/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/offline_calibration/sub-01/ses-002/run1_online_gaze2D.npz'

movies = ['/home/labopb/Documents/Marie/neuromod/Mario/Eye-tracking/pupil_data/sub-01/ses-002/sub-01_ses-002_20210611-105659_SuperMarioBros-Nes_Level2-3_000.bk2',
          '',
          '']

if online:
    # Convert serialized file to list of dictionaries...
    seri_gaze = load_pldata_file(sgaze_path, 'gaze')[0]
    deserialized_gaze = []

    for gaze in seri_gaze:
        gaze_data = {}
        for key in gaze.keys():
            if key != 'base_data':
                gaze_data[key] = gaze[key]
        deserialized_gaze.append(gaze_data)

    np.savez(args.outfile, gaze2d = deserialized_gaze)

else:
    gaze = np.load(gaze_path, allow_pickle=True)
