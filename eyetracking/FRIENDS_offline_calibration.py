import os, sys, platform, json
from time import time
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace

import argparse


def get_arguments():

    parser = argparse.ArgumentParser(description='Perform off-line gaze mapping with 2D and 3D pupil detectors ')
    parser.add_argument('--run_dir', default='', type=str, help='absolute path to main code directory')
    parser.add_argument('--config', default='config.json', type=str, help='absolute path to config file')
    args = parser.parse_args()

    return args


def make_intrinsics(file_path):
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

    save_object(int_dict, file_path)
    '''
    pass


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


def predict_all_those_pupils(config, pd_2d, eye_file, pd_3d=None, label='test'):

    num_frames = len(eye_file.timestamps)

    pupils_2d = []
    pupils_3d = []

    for i in tqdm(range(num_frames), desc = 'Predicting Pupil Positions'):
        try:
            frame = eye_file.get_frame()

            predicted2d = pd_2d.detect(frame)
            pupils_2d.append(predicted2d)

            if config['use3D']:
                predicted3d = pd_3d.detect(frame, previous_detection_results = [predicted2d])
                pupils_3d.append(predicted3d)

        except:
            print("Prediction failed on frame " + str(i))

    pupils_2d_np = np.array(pupils_2d)
    np.savez(os.path.join(config['out_dir'], label+'_pupils2D.npz'), pupils2d = pupils_2d_np)

    try:
        # TODO: specify if calibration or run's pupils in name of saved file
        p2d_writer = PLData_Writer(directory=config['out_dir'], name=label+'_offline_pupil2d')
        p2d_writer.extend(pupils_2d)
        p2d_writer.close()
    except:
        print('Could not export run 2d pupils')

    if config['use3D']:
        pupils_3d_np = np.array(pupils_3d)
        np.savez(os.path.join(config['out_dir'], label+'_pupils3D.npz'), pupils3d = pupils_3d_np)
        try:
            # TODO: specify if calibration or run's pupils in name of saved file
            p3d_writer = PLData_Writer(directory=config['out_dir'], name=label+'_offline_pupil3d')
            p3d_writer.extend(pupils_3d)
            p3d_writer.close()
        except:
            print('Could not export run 2d pupils')

    return pupils_2d, pupils_3d


def map_all_those_pupils_to_gaze(config, gm_2d, gm_3d, pupils_2d, pupils_3d, label='test'):

    gaze_data_2d = gm_2d.map_pupil_to_gaze(pupils_2d, sort_by_creation_time=True)

    gaze_2d = []
    gaze_3d = []

    for i in tqdm(gaze_data_2d, desc = 'Mapping 2D Pupils to Gaze'):
        gaze_2d.append(i)

    gaze_2d_np = np.array(gaze_2d)
    np.savez(os.path.join(config['out_dir'], label+'_gaze2D.npz'), gaze2d = gaze_2d)

    if config['use3D']:
        gaze_data_3d = gm_3d.map_pupil_to_gaze(pupils_3d, sort_by_creation_time=True)

        for j in tqdm(gaze_data_3d, desc = 'Mapping 3D Pupils to Gaze'):
            gaze_3d.append(j)

        gaze_3d_np = np.array(gaze_3d)
        np.savez(os.path.join(config['out_dir'], label+'_gaze3D.npz'), gaze3d = gaze_3d)

    return gaze_2d, gaze_3d


if __name__ == '__main__':
    '''
    Script takes pupil outputs from Friends s2e04 (a and b; two runs) seen in the scanner while collecting
    eyetracking data.

    These gold standard gaze mappings can be contrasted with Deepgaze_MR predictions
    Project repo here: https://github.com/courtois-neuromod/deepgaze_mr
    '''
    args = get_arguments()

    sys.path.append(os.path.join(args.run_dir, "pupil", "pupil_src", "shared_modules"))
    from video_capture.file_backend import File_Source
    from file_methods import PLData_Writer, load_pldata_file, load_object, save_object
    from gaze_producer.worker.fake_gpool import FakeGPool, FakeIPC

    from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
    from gaze_mapping.gazer_2d import Gazer2D

    from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
    from gaze_mapping.gazer_3d.gazer_headset import Gazer3D


    with open(args.config, 'r') as f:
        cfg = json.load(f)

    detect_2d = None
    detect_3d = None
    gazemap_2d = None
    gazemap_3d = None

    cal_gazemap_2d = None
    cal_gazemap_3d = None

    g_pool = make_detection_gpool()

    '''
    Step 1. Detect calibration pupils offline with 2D and 3D pupil detectors
    This step is optional: the pupil data calculated online (during scan) can be used instead
    (those are exported as .npz file by psychopy;
    technically, pupil's own .pldata files could be used, but they won't load...)
    '''

    if cfg['overwrite_camera_intrinsics']:
        '''
        TODO: generate new eye0.instrinsics file to replace the existing one...
        make_intrinsics(cfg['calib_mp4'][:-4]+'.instrinsics')
        '''
        pass

    # Instantiate pupil detectors
    # This step is needed for gpool ReGARDLESS of whether pupil detection
    # is re-done offline or not
    calib_eye_file = File_Source(g_pool, source_path=cfg['calib_mp4'])

    detect_2d = Detector2DPlugin(g_pool, cfg['properties'])
    if cfg['use3D']:
        detect_3d = Pye3DPlugin(g_pool)

    # Predict pupils offline
    if cfg['detect_calib_pupils']:
        calib_pupils_2d, calib_pupils_3d = predict_all_those_pupils(cfg, detect_2d, calib_eye_file, detect_3d, 'calib')


    '''
    Step 2. Initialize the 2d and 3d Gazer models
    '''
    #g_pool.capture.intrinsics.focal_length = cfg['focal_length']
    g_pool.min_calibration_confidence = cfg['min_calibration_confidence']
    g_pool.ipc_pub = FakeIPC()
    g_pool.get_timestamp = time

    # Option A: use existing calibration parameters (.plcal file)
    if cfg['use_online_calib_params']:

        cal_path_2d = cfg['calibration_parameters_2d']
        cal_params_2d  = load_object(cal_path_2d, allow_legacy=True)['data']['calib_params']
        gazemap_2d = Gazer2D(g_pool, params=cal_params_2d)

        if cfg['export_calib_gaze']:
            cal_gazemap_2d = Gazer2D(g_pool, params=cal_params_2d)
        # Note: in our set up, online calibration is always w 2D model
        # but if 3d offline calibration params are saved, they can be uploaded like this
        if cfg['use3D']:
            cal_path_3d = cfg['calibration_parameters_3d']
            cal_params_3d  = load_object(cal_path_3d, allow_legacy=True)['data']['calib_params']
            gazemap_3d = Gazer3D(g_pool, params=cal_params_3d)

            if cfg['export_calib_gaze']:
                cal_gazemap_3d = Gazer3D(g_pool, params=cal_params_3d)

    # Option B: use marker and pupil data to fit a gazer model
    # Pupil data predicted online during calibration can be used from saved .pnz file,
    # or the pupils predicted with the optional step above can be used instead
    else:
        cal_path = cfg['calibration_data']
        cal_file = np.load(cal_path, allow_pickle=True)

        cal_data = {}
        cal_data['ref_list'] = cal_file['markers'].tolist()

        # load pupils detected online if no offline calibration pupils produced from previous optional step
        if not cfg['detect_calib_pupils']:
            # use pupils from Basile's file (better organized)
            calib_pupils_2d = cal_file['pupils'].tolist()

            '''
            # Alternatively, use software's own pupil file (.plcal), either outputed offline or online
            calib_pupils_2d = []

            # Convert serialized file to list of dictionaries...
            #seri_calib_pupils_2d = load_pldata_file(cfg['calib_mp4'][:-9], 'pupil')
            seri_calib_pupils_2d = load_pldata_file(cfg['previous_cal_2dpupils'], cfg['prev_cal_pup2D_name'])
            for pup in seri_calib_pupils_2d[0]:
                pupil_data = {}
                for key in pup.keys():
                    if key == 'ellipse':
                        pupil_data[key] = {}
                        for sub_key in pup[key]:
                            pupil_data[key][sub_key] = pup[key][sub_key]
                    else:
                        pupil_data[key] = pup[key]
                calib_pupils_2d.append(pupil_data)

            '''
            calib_pupils_3d = []

            if cfg['use3D']:
                # Convert serialized file to list of dictionaries...
                seri_calib_pupils_3d = load_pldata_file(cfg['previous_cal_3dpupils'], cfg['prev_cal_pup3D_name'])
                for pup in seri_calib_pupils_3d[0]:
                    pupil_data = {}
                    for key in pup.keys():
                        if key == 'ellipse':
                            pupil_data[key] = {}
                            for sub_key in pup[key]:
                                pupil_data[key][sub_key] = pup[key][sub_key]
                        else:
                            pupil_data[key] = pup[key]
                    calib_pupils_3d.append(pupil_data)


        cal_data['pupil_list'] = calib_pupils_2d

        # initialization calls Gazer_base's method fit_on_calib_data() to fit calibration parameters from markers and pupils
        gazemap_2d = Gazer2D(g_pool, calib_data=cal_data)
        # sanity check
        assert(gazemap_2d.right_model._is_fitted == True)
        assert(gazemap_2d.binocular_model._is_fitted == False)
        assert(gazemap_2d.left_model._is_fitted == False)

        if cfg['export_calib_gaze']:
            cal_gazemap_2d = Gazer2D(g_pool, calib_data=cal_data)
        # export newly created calibration parameters as .plcal file
        #cal_params = gazemap_2d.get_params()
        cal_params = {}
        cal_params['data'] = {}
        cal_params['data']['calib_params'] = gazemap_2d.get_params()
        save_object(cal_params, os.path.join(cfg['out_dir'], 'Offline_Calibration2D.plcal'))

        if cfg['use3D']:
            cal_data['pupil_list'] = calib_pupils_3d

            gazemap_3d = Gazer3D(g_pool, calib_data=cal_data)

            if cfg['export_calib_gaze']:
                cal_gazemap_3d = Gazer3D(g_pool, calib_data=cal_data)

            assert(gazemap_3d.right_model._is_fitted == True)
            assert(gazemap_3d.binocular_model._is_fitted == False)
            assert(gazemap_3d.left_model._is_fitted == False)

            # export newly created calibration parameters as .plcal file
            cal_params = {}
            cal_params['data'] = {}
            cal_params['data']['calib_params'] = gazemap_3d.get_params()
            save_object(cal_params, os.path.join(cfg['out_dir'], 'Offline_Calibration3D.plcal'))

    if cfg['export_calib_gaze']:
        # map calibration pupils to gaze
        calib_gaze_2d, calib_gaze_3d = map_all_those_pupils_to_gaze(cfg, cal_gazemap_2d, cal_gazemap_3d, calib_pupils_2d, calib_pupils_3d, 'calib')


    '''
    Step 3. Predict run's pupils and map those pupils to gaze position
    '''
    # predict run's pupils offline
    if cfg['detect_run_pupils']:

        if cfg['overwrite_camera_intrinsics']:
            '''
            TODO: generate new eye0.instrinsics file...
            make_intrinsics(cfg['run_mp4'][:-4]+'.instrinsics')
            '''
            pass

        run_eye_file = File_Source(g_pool, source_path=cfg['run_mp4'])
        # overwrite dummy variable set when loading intrinsics file...
        #run_eye_file._intrinsics.focal_length = cfg['focal_length']
        #g_pool.capture.intrinsics.focal_length = cfg['focal_length']

        if cfg['use3D']:

            if cfg['freeze_3d_at_test']:
                detect_3d.detector._long_term_schedule.pause()
                detect_3d.detector._ult_long_term_schedule.pause()

        # Predict run's pupils
        run_pupils_2d, run_pupils_3d = predict_all_those_pupils(cfg, detect_2d, run_eye_file, detect_3d, 'run_'+cfg['run_num'])

    # load run's online pupils
    else:
        seri_run_pupils_2d = load_pldata_file(cfg['previous_run_2dpupils'], cfg['prev_run_pup2D_name'])
        #seri_run_pupils_2d = load_pldata_file(cfg['run_mp4'][:-9], 'pupil')
        #run_gaze_2d = load_pldata_file(cfg['run_mp4'][:-9], 'gaze')

        # Convert serialized file to list of dictionaries...
        run_pupils_2d = []
        for pup in seri_run_pupils_2d[0]:
            pupil_data = {}
            for key in pup.keys():
                if key == 'ellipse':
                    pupil_data[key] = {}
                    for sub_key in pup[key]:
                        pupil_data[key][sub_key] = pup[key][sub_key]
                else:
                    pupil_data[key] = pup[key]
                #pupil_data[key] = pup[key]
            run_pupils_2d.append(pupil_data)

        # https://github.com/pupil-labs/pupil/blob/97a3d099c2ffe353d0d1534ebde45ac0e1145da0/pupil_src/shared_modules/file_methods.py#L143

        # Note: there are no 3d online pupils, cannot use cfg['detect_run_pupils']=True and cfg['use3D']=True
        run_pupils_3d = []

        if cfg['use3D']:
            # Convert serialized file to list of dictionaries...
            seri_run_pupils_3d = load_pldata_file(cfg['previous_run_3dpupils'], cfg['prev_run_pup3D_name'])
            for pup in seri_run_pupils_3d[0]:
                pupil_data = {}
                for key in pup.keys():
                    if key == 'ellipse':
                        pupil_data[key] = {}
                        for sub_key in pup[key]:
                            pupil_data[key][sub_key] = pup[key][sub_key]
                    else:
                        pupil_data[key] = pup[key]
                    #pupil_data[key] = pup[key]
                run_pupils_3d.append(pupil_data)

    # map run pupils to gaze
    gaze_data_2d, gaze_data_3d = map_all_those_pupils_to_gaze(cfg, gazemap_2d, gazemap_3d, run_pupils_2d, run_pupils_3d, 'run_'+cfg['run_num'])
