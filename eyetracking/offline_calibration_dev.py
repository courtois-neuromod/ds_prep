'''
Script 1: Offline 2D and 3D Calibration

Purpose:
    Use PsychoPy calibration markers (saved as .npz) to do offline calibration and output offline .plcal calibration file

Steps:
    1. Pupil detection:
        1.1 read eye0.mp4 files using the FileBackend
        1.2 Instantiate 2d Detector and 3d detector
        1.3 Loop over frames feeding them to the 2d detector predict function,
        then feeding the frame and 2d pupil to the 3d detector predict function, accumulate predictions.
        Quality Check: visualize pupil detections on eye frames (see notebook)
    2. Gaze Mapping:
        2.1 Instantiate Gazer2d with either
            A. online calibration parameters (.pcal file), or
            B. marker and pupil data (.npz) exported with psychopy during calibration;
             Fit model params from data and export new offline calibration (.pcal)
        https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/gaze_mapping/gazer_2d.py#L168
        2.2 Map all pupils to gaze with Gazer2D
        2.3 Instantiate Gazer3d...
        https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/gaze_mapping/gazer_3d/gazer_headset.py#L327
        2.4 Map all pupils to gaze with Gazer2D


Note: Might require using mock g_pool. It is apparently also used in some parts of pupil
https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/gaze_producer/worker/fake_gpool.py
'''

import os, sys, platform, json
#import logging
from time import time
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace

sys.path.append(os.path.join("/home/labopb/Documents/Marie/neuromod/pupil", "pupil_src", "shared_modules"))
#from pupil_src.shared_modules.video_capture.file_backend import File_Source
#from pupil_src.shared_modules.file_methods import load_object
#from pupil_src.shared_modules.gaze_producer.worker.fake_gpool import FakeGPool, FakeIPC
from video_capture.file_backend import File_Source
from file_methods import load_object, save_object
from gaze_producer.worker.fake_gpool import FakeGPool, FakeIPC


import argparse

parser = argparse.ArgumentParser(description='Perform off-line gaze mapping with 2D and 3D pupil detectors ')
parser.add_argument('--run_dir', default='', type=str, help='absolute path to main code directory')
parser.add_argument('--config', default='config.json', type=str, help='absolute path to config file')
#parser.add_argument('--model', default='detect2d', type=str, choices=['detect2d', 'detect3d'])
parser.add_argument('--use3D', action='store_true', default=False, help='Use 3D pupil detector')
args = parser.parse_args()

#from pupil_src.shared_modules.pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
#if args.use3D:
#    from pupil_src.shared_modules.pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
if args.use3D:
    from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin


from gaze_mapping.gazer_2d import Gazer2D
#from gaze_mapping.gazer_base import GazerBase

if __name__ == "__main__":
    '''
    STEP 1.1
    Read eye0.mp4 files using the FileBackend
    https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/video_capture/file_backend.py#L202

    Their demo file
    mp4_path = '/home/labopb/Documents/Marie/neuromod/test_data/pupil/Sample_pupil_data/sample_recording_v2/eye0.mp4'
    One of our files
    mp4_path = '/home/labopb/Documents/Marie/neuromod/test_data/pupil/THINGS_eye_tracking/ses-001/sub-03_ses-001_20210421-131339.pupil/task-thingsmemory_run-2/000/eye0.mp4'
    '''

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    mp4_path = cfg['calib_mp4']

    '''
    Specify g_pool features called by 2d / 3d pupil detectors (iteration and detect)
    Iterating File_Source object requires a g_pool file to store/update/share common info

    As reference: specs from our Psychopy eyetracking task (cneuromod)
    https://github.com/courtois-neuromod/task_stimuli/blob/master/src/shared/eyetracking.py
    '''

    # blank g_pool placeholder
    g_pool = SimpleNamespace()

    '''
    # UPDATE: all capture attributes are loaded from the camera attributes from eye0.intrinsics
    No need to specify them manually

    int = SimpleNamespace()
    #int.focal_length = 1000.0 # (DUMMY VARIABLE, in PIXELS) TODO: determine camera focal lenght IF NEEDED by 3d pupil detector...
    int.focal_length = 256.0 # ?? 16mm * 640 pix / ~40mm ? ~= 256.0

    #F(pixels) = F(mm) * ImageWidth (pixel) / SensorWidth(mm) ~ 6mm * 640 / FOV width (where FOV width is in mm)
    # 16mm * 640 pix / ~40mm ? ~= 256.0

    # focal_pixel = (image_width_in_pixels * 0.5) / tan(FOV * 0.5 * PI/180) where FOV width is in degrees
    # (640 pix * 0.5) / math.tan (103 deg FOV * 0.5 * (math.pi/180) ) = 254.0
    # 103 deg FOV here: https://docs.pupil-labs.com/core/software/pupil-capture/#camera-intrinsics-estimation

    #F(mm) = F(pixels) * SensorWidth(mm) / ImageWidth (pixel)
    #
    #https://www.mrc-systems.de/en/products/mr-compatible-cameras#camera-mirror-mounts
    #https://www.mrc-systems.de/downloads/en/mri-compatible-cameras/MRCam_eye-tracking_brochure.pdf
    #MRC HiSpeed camera (comparable to ours): 4.6 mm, 6 mm, 8 mm
    #ours: MRC Systems GmbH-GVRD-MRC HighSpeed-MR_CAM_HS_001
    int.resolution = (640, 480) # corresponds to spatial res, NOT temporal

    cap = SimpleNamespace()
    cap.frame_size =  (640, 480) # (width, height)
    # cap.frame_rate: 250  # No need to specify frame rate: the value is derived from saved time stamps from eye0.mp4 file; 0.0040157 ~ 1/250
    # https://github.com/pupil-labs/pupil/blob/97a3d099c2ffe353d0d1534ebde45ac0e1145da0/pupil_src/shared_modules/video_capture/file_backend.py#L245
    cap.intrinsics = int # Note: dummy camera intrinsics currently loaded w our eye0.mp4 files...
    g_pool.capture = cap

    Tests: load intrinsics file "eye0.intrinsics" used by script to upload saved camera specs...
    from pupil_src.shared_modules.camera_models import Camera_Model

    intrinsics = Camera_Model.from_file()
    file_path = '/home/labopb/Documents/Marie/neuromod/test_data/pupil/THINGS_eye_tracking/ses-001/sub-03_ses-001_20210421-131339.pupil/task-thingsmemory_run-2/000/eye0.intrinsics'
    intrinsics_dict = load_object(file_path, allow_legacy=False)
    # the file's only keys are version and resolution... for 3d model, also need camera FOCAL LENGHT?

    '''

    rbounds = SimpleNamespace()
    rbounds.bounds = (0, 0, 640, 480) # (minx, miny, maxx, maxy) # TODO: optimize? Narrow down search window?
    g_pool.roi = rbounds

    g_pool.display_mode = "algorithm" # "roi" # for display; doesn't change much
    g_pool.eye_id = 0 #'eye0'
    g_pool.app = "player" # "capture"

    '''
    TODO: recalculate camera intrinsics
    **Please recalculate the camera intrinsics using the Camera Intrinsics Estimation."
    Doc link: https://docs.pupil-labs.com/core/software/pupil-capture/#camera-intrinsics-estimation
    int_dict = {'version': 1,
                'resolution': [640, 480],
                'cam_type': 'dummy',
                'camera_matrix': [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]],
                'dist_coefs': [[0.0, 0.0, 0.0, 0.0, 0.0]],
                'focal_length': 256.0
    }

    save_object(int_dict, mp4_path[:-4]+'.intrinsics')
    '''

    eye_file = File_Source(g_pool, source_path=mp4_path)

    num_frames = len(eye_file.timestamps) # these should be equivalent

    '''
    Comparison of eye_file features, # theirs, ours
    eye_file.timing # own, own
    eye_file._frame_rate # 0.006231420019873308, 0.004015749212087875 (~1/250 where 250 is fps)
    eye_file._intrinsics.resolution # (192, 192), (640, 480)
    eye_file._intrinsics.name # eye0, eye0
    eye_file._intrinsics.focal_length # 283.269172, 1000.0 # Is ours a DUMMY Variable?
    eye_file.timestamps # len = 16104, 79070
    '''



    '''
    STEP 1.2
    Instantiate 2d Detector and 3d Detector
    # https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py#L40
    # https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/pupil_detector_plugins/pye3d_plugin.py#L36

    Reference for 3d model (uses 3 simultaneous time frames, can be frozen or not)
    Documentation: https://docs.pupil-labs.com/developer/core/pye3d/
    Source Code: https://github.com/pupil-labs/pye3d-detector/
    '''

    #properties = {"intensity_range": 4} # As specified in psychopy script; play w parameters?
    properties = cfg['properties']

    '''Default properties values
    properties = {'coarse_detection': True,
                  'coarse_filter_min': 128,
                  'coarse_filter_max': 280,
                  'intensity_range': 23,
                  'blur_size': 5,
                  'canny_treshold': 160,
                  'canny_ration': 2,
                  'canny_aperture': 5,
                  'pupil_size_max': 100,
                  'pupil_size_min': 10,
                  'strong_perimeter_ratio_range_min': 0.8,
                  'strong_perimeter_ratio_range_max': 1.1,
                  'strong_area_ratio_range_min': 0.6,
                  'strong_area_ratio_range_max': 1.1,
                  'contour_size_min': 5,
                  'ellipse_roundness_ratio': 0.1,
                  'initial_ellipse_fit_treshhold': 1.8,
                  'final_perimeter_ratio_range_min': 0.6,
                  'final_perimeter_ratio_range_max': 1.2,
                  'ellipse_true_support_min_dist': 2.5,
                  'support_pixel_ratio_exponent': 2.0}
    '''

    detect_2d = Detector2DPlugin(g_pool, properties)

    if args.use3D:
        detect_3d = Pye3DPlugin(g_pool)

        '''
        Functions to freeze / unfreeze Pye3DPlugin (3d pupil detector)
        https://github.com/pupil-labs/pye3d-detector/blob/e7f470ade72704da154a29fecb540625c645d84f/pye3d/detector_3d.py#L160

        # to determine if frozen
        detect_3d.detector._long_term_schedule.is_paused
        # to freeze (unfrozen by default)
        detect_3d.detector._long_term_schedule.pause()
        detect_3d.detector._ult_long_term_schedule.pause()
        # to unfreeze
        detect_3d.detector._long_term_schedule.resume()
        detect_3d.detector._ult_long_term_schedule.resume()
        '''

    '''
    STEP 1.3
    Loop over frames feeding them to the 2d detector predict() function,
    then feeding the frame and 2d pupil to the 3d detector predict() function, accumulate predictions.

    TODO: determine in what format to export pupil predictions (.pldata or .npz)
    # .pldata needed for qc and visualization in pupil player interface...

    From our psychopy output: pupils and calibration markers are saved as .npz file w two entries: pupils and markers
    pp_output = np.load('path/to/file.npz', allow_pickle=True)
    pp_output.files # ['pupils', 'markers']

    pp_output['pupils']
    Note: pupils is a 1D numpy array of time (num_frames, ) with each entry a dictionary of predictions outputed by pupil detector's "detect" functions (for each frame)

    pp_output['markers']
    Note: markers is a 1D numpy array of time (num_frames, ) with each entry a dictionary of coordinates for the calibration marker
    It contains the normalized position, the screen position in pixels, and the marker's time stamp (for that frame)

    np.savez('path/to/file.npz', pupils=pp_output['pupils'], markers=pp_output['markers'])

    Note however that pupil exports predictions as .pldata files...
    https://docs.pupil-labs.com/developer/core/recording-format/
    "Pupil Player decodes the messages into file_methods.Serialized_Dicts.
    Each Serialized_Dict instance holds the serialized second frame and is responsible for decoding it
    on demand. The class is designed such that there is a maximum number of decoded frames at the same time.
    This prevents Pupil Player from using excessive amounts of memory.

    You can use file_methods.PLData_Writer and file_methods.load_pldata_file() to read and write pldata files."

    To load file:
    pupils_data = load_pldata_file('/path/to/directory', topic='pupil')
    '''

    pupils_2d = []

    if args.use3D:
        pupils_3d = []

    for i in tqdm(range(num_frames), desc = 'Predicting Pupil Positions'):
        try:
            # returns Frame object, which contains an _img attribute (numpy array) that is a frame of the eye movie
            # https://github.com/pupil-labs/pupil/blob/97a3d099c2ffe353d0d1534ebde45ac0e1145da0/pupil_src/shared_modules/video_capture/file_backend.py#L48
            frame = eye_file.get_frame()
            #frame.img, frame.timestamp, frame.index, frame.width, frame.height
            #eye_file.target_frame_idx

            # 2d model prediction; output is a dictionary
            # normalized position x (width), y (height) = pixel value divided by image size
            # NOTE that y IS FLIPPED so that high normalized values are high in the image (top)
            predicted2d = detect_2d.detect(frame)
            pupils_2d.append(predicted2d)

            # 3d model prediction; output is a dictionary
            if args.use3D:
                predicted3d = detect_3d.detect(frame, previous_detection_results = [predicted2d])
                pupils_3d.append(predicted3d)

        except:
            print("Prediction failed on frame " + str(i))



    out_dir = cfg['out_dir']
    pupils_2d_np = np.array(pupils_2d)
    np.savez(os.path.join(out_dir, '2d_pupils.npz'), pupils2d = pupils_2d_np)
    try:
        save_object(pupils_2d_np, os.path.join(out_dir, 'pupil.pldata'))
        '''
        Note : the file exports fine, but this line crashes when I try to load these pupils directly in the player interface...
        https://github.com/pupil-labs/pupil/blob/97a3d099c2ffe353d0d1534ebde45ac0e1145da0/pupil_src/shared_modules/file_methods.py#L143
        '''
    except:
        print('Could not export 2d pupil data as pupil.pldata file')

    if args.use3D:
        pupils_3d_np = np.array(pupils_3d)
        np.savez(os.path.join(out_dir, '3d_pupils.npz'), pupils3d = pupils_3d_np)
        try:
            save_object(pupils_2d_np, os.path.join(out_dir, 'pupil3d.pldata'))
        except:
            print('Could not export 3d pupil data as pupil3d.pldata file')



    '''
    STEP 2.1
    Instantiate and initialize Gazer2d
    https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/gaze_mapping/gazer_2d.py#L168
    https://github.com/pupil-labs/pupil/blob/97a3d099c2ffe353d0d1534ebde45ac0e1145da0/pupil_src/shared_modules/gaze_mapping/gazer_base.py#L267

    Option A: instantiate with existing calibration parameters
    Load existing calibration file .plcal with load_file()
    No need to fit on calibration data: parameters already fitted
    (.pcal is the format in which new offline calibrations need to be exported to be used by gaze mapping models...)

    Option B: instantiate with pupils and markers data to generate offline calibration
    Load the calibration markers (.npz file) exported from psychopy to get the markers positions/timings
    The file contains "pupils" and "markers" data
    Either use the data as is, or replace the "pupils" data (predicted online) with
    the new pupils predicted offline with the 2d or 3d model in Step 1

    Fit calibration parameters on calib data (from .npz), and save new offline calibration file (.pcal)
    '''
    g_pool.min_calibration_confidence = 0.6 # TODO: pick good value... link below has default of 0.6 confidence
    # https://github.com/pupil-labs/pupil/blob/97a3d099c2ffe353d0d1534ebde45ac0e1145da0/pupil_src/shared_modules/gaze_mapping/matching.py#L19

    g_pool.ipc_pub = FakeIPC() # has notify(notification) method called when initiating the gazer model
    g_pool.get_timestamp = time

    # Option A, initialize with existing calibration (.plcal file)
    if 'calibration_parameters' in cfg:
        cal_path = cfg['calibration_parameters']
        cal_params  = load_object(cal_path, allow_legacy=True)['data']['calib_params']
        # note: params split between right, left and binoc models;
        # 2d params: coef_ (list of two lists of 6 floats) and intercept_ (list of two floats)
        # 3d params: eye_camera_to_world_matrix (list of 4 lists of 4 floats) and gaze_distance (float)

        if args.use3D:
            pass
            #gazemap_3d = ???
        else:
            gazemap_2d = Gazer2D(g_pool, params=cal_params)

    # Option B, fit calibration parameters from loaded marker and pupil data
    elif 'calibration_data' in cfg:
        cal_path = cfg['calibration_data']
        cal_file = np.load(cal_path, allow_pickle=True)

        # fit_on_calib_data function takes a dictionary with two entries: "ref_list" and "pupil_list", both lists of dictionaries
        # https://github.com/pupil-labs/pupil/blob/97a3d099c2ffe353d0d1534ebde45ac0e1145da0/pupil_src/shared_modules/gaze_mapping/gazer_base.py#L267
        cal_data = {}
        cal_data['ref_list'] = cal_file['markers'].tolist()
        cal_data['pupil_list'] = cal_file['pupils'].tolist()

        if args.use3D:
            pass
            #gazemap_3d = ???
        else:
            gazemap_2d = Gazer2D(g_pool, calib_data=cal_data)
            # initialization calls Gazer_base's method fit_on_calib_data() to fit calibration parameters from markers and pupils
            assert(gazemap_2d.right_model._is_fitted == True)
            assert(gazemap_2d.binocular_model._is_fitted == False)
            assert(gazemap_2d.left_model._is_fitted == False)

            # export newly created calibration calibration parameters
            cal_params = gazemap_2d.get_params()
            save_object(cal_params, os.path.join(out_dir, 'Offline_Calibration.plcal'))

    '''
    STEP 2.2 Map all pupils to gaze position
    '''
    # load pupil data: list of dictionaries, one per pupil
    pupil_data = pupils_2d # from step 1
    #pupil_data = pupils_3d
    gaze_data_2d = gazemap_2d.map_pupil_to_gaze(pupil_data, sort_by_creation_time=True)

    gaze_2d = []
    for i in gaze_data_2d:
        gaze_2d.append(i)

    out_dir = cfg['out_dir']
    gaze_2d_np = np.array(gaze_2d)
    np.savez(os.path.join(out_dir, '2d_gaze.npz'), gaze2d = gaze_2d)
