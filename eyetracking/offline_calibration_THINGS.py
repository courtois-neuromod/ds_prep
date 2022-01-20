import os, sys, platform, json
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from types import SimpleNamespace

import argparse

from quality_check_THINGS import assess_timegaps, qc_report

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


def map_run_gaze(cfg, run):
    if cfg['apply_qc']:
        gct = str(cfg['gaze_confidence_threshold'])
        pct = str(cfg['pupil_confidence_threshold'])

        gaze_report = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + gct + ' Confidence Threshold', 'Outside Screen Area'])
        pupil_report = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + pct + ' Confidence Threshold'])

        run_report = open(os.path.join(cfg['out_dir'], 'qc', 'run' + run + '_report.txt'), 'w+')

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
    # This step is needed to initialize a valid gpool file ReGARDLESS of whether pupil detection
    # is re-done offline or not
    calib_eye_file = File_Source(g_pool, source_path=cfg['run' + run + '_calib_mp4'])

    # run QC to detect missing frames in eye movie (e.g., camera freeze)
    if cfg['apply_qc']:
        calib_t_stamps = calib_eye_file.timestamps
        diff_list, gap_idx = assess_timegaps(calib_t_stamps, cfg['time_threshold'])
        if len(gap_idx) > 0:
            #export as .tsv
            np.savetxt(os.path.join(cfg['out_dir'], 'qc', 'run' + run + '_calib_framegaps.tsv'), np.array(gap_idx), delimiter="\t")

    # Initialize 2d pupil detector
    detect_2d = Detector2DPlugin(g_pool, cfg['properties'])
    if cfg['use3D']:
        detect_3d = Pye3DPlugin(g_pool)

    # Predict pupils offline
    if cfg['detect_calib_pupils']:
        calib_pupils_2d, calib_pupils_3d = predict_all_those_pupils(cfg, detect_2d, calib_eye_file, detect_3d, 'run' + run + '_calib')

        # QC the offline pupils
        if cfg['apply_qc']:
            cp_off2d_s, cp_off2d_d = qc_report(calib_pupils_2d, cfg['out_dir'] + '/qc', 'pupil_calib_offline2D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
            run_report.write(cp_off2d_s + '\n')
            cp_off2d_d = [cp_off2d_d[0], 'Calib', 'Offline2D', 'Run' + run, cp_off2d_d[1]]
            pupil_report = pupil_report.append(pd.Series(cp_off2d_d, index=pupil_report.columns), ignore_index=True)
            if cfg['use3D']:
                cp_off3d_s, cp_off3d_d  = qc_report(calib_pupils_3d, cfg['out_dir'] + '/qc', 'pupil_calib_offline3D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
                run_report.write(cp_off3d_s + '\n')
                cp_off3d_d = [cp_off3d_d[0], 'Calib', 'Offline3D', 'Run' + run, cp_off3d_d[1]]
                pupil_report = pupil_report.append(pd.Series(cp_off3d_d, index=pupil_report.columns), ignore_index=True)

    # QC the online pupils
    if cfg['apply_qc']:
        calib_online_pupils = load_pldata_file(cfg['run' + run + '_calib_mp4'][:-9], 'pupil')[0]
        cp_on2d_s, cp_on2d_d = qc_report(calib_online_pupils, cfg['out_dir'] + '/qc', 'pupil_calib_online2D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
        run_report.write(cp_on2d_s + '\n')
        cp_on2d_d = [cp_on2d_d[0], 'Calib', 'Online2D', 'Run' + run, cp_on2d_d[1]]
        pupil_report = pupil_report.append(pd.Series(cp_on2d_d, index=pupil_report.columns), ignore_index=True)

    '''
    Step 2. Initialize the 2d and 3d Gazer models
    '''
    #g_pool.capture.intrinsics.focal_length = cfg['focal_length']
    g_pool.min_calibration_confidence = cfg['min_calibration_confidence']
    g_pool.ipc_pub = FakeIPC()
    g_pool.get_timestamp = time

    # Option A: use existing calibration parameters (.plcal file)
    if cfg['use_online_calib_params']:

        cal_path_2d = cfg['run' + run + '_calibration_parameters_2d']
        cal_params_2d  = load_object(cal_path_2d, allow_legacy=True)['data']['calib_params']
        gazemap_2d = Gazer2D(g_pool, params=cal_params_2d)

        # Instantiate a separate gaze mapper for the calibration pupil,
        # different from the one applied to run pupils (in case they interfere?)
        if cfg['export_calib_gaze']:
            cal_gazemap_2d = Gazer2D(g_pool, params=cal_params_2d)
        # Note: in our set up, online calibration is always w 2D model
        # but if 3d offline calibration params are saved, they can be uploaded like this
        if cfg['use3D']:
            cal_path_3d = cfg['run' + run + '_calibration_parameters_3d']
            cal_params_3d  = load_object(cal_path_3d, allow_legacy=True)['data']['calib_params']
            gazemap_3d = Gazer3D(g_pool, params=cal_params_3d)

            if cfg['export_calib_gaze']:
                cal_gazemap_3d = Gazer3D(g_pool, params=cal_params_3d)

    # Option B: use marker and pupil data to fit a gazer model
    # Pupil data predicted online during calibration can be used from saved .pnz file,
    # or the pupils predicted with the optional step above can be used instead
    else:
        cal_path = cfg['run' + run + '_calibration_data']
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
                seri_calib_pupils_3d = load_pldata_file(cfg['run' + run + '_previous_cal_3dpupils'], cfg['run' + run + '_prev_cal_pup3D_name'])
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
        save_object(cal_params, os.path.join(cfg['out_dir'], 'Offline_Calibration2D_run' + run + '.plcal'))

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
            save_object(cal_params, os.path.join(cfg['out_dir'], 'Offline_Calibration3D_run' + run + '.plcal'))

    if cfg['export_calib_gaze']:
        # map calibration pupils to gaze
        calib_gaze_2d, calib_gaze_3d = map_all_those_pupils_to_gaze(cfg, cal_gazemap_2d, cal_gazemap_3d, calib_pupils_2d, calib_pupils_3d, 'run' + run + '_calib')

        # QC offline gaze
        if cfg['apply_qc']:
            cg_off2d_s, cg_off2d_d = qc_report(calib_gaze_2d, cfg['out_dir'] + '/qc', 'gaze_calib_offline2D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
            run_report.write(cg_off2d_s + '\n')
            cg_off2d_d = [cg_off2d_d[0], 'Calib', 'Offline2D', 'Run' + run, cg_off2d_d[1], cg_off2d_d[2]]
            gaze_report = gaze_report.append(pd.Series(cg_off2d_d, index=gaze_report.columns), ignore_index=True)
            if cfg['use3D']:
                cg_off3d_s, cg_off3d_d = qc_report(calib_gaze_3d, cfg['out_dir'] + '/qc', 'gaze_calib_offline3D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
                run_report.write(cg_off3d_s + '\n')
                cg_off3d_d = [cg_off3d_d[0], 'Calib', 'Offline3D', 'Run' + run, cg_off3d_d[1], cg_off3d_d[2]]
                gaze_report = gaze_report.append(pd.Series(cg_off3d_d, index=gaze_report.columns), ignore_index=True)

    # QC online gaze
    if cfg['apply_qc']:
        calib_online_gaze = load_pldata_file(cfg['run' + run + '_calib_mp4'][:-9], 'gaze')[0]
        cg_on2d_s, cg_on2d_d = qc_report(calib_online_gaze, cfg['out_dir'] + '/qc', 'gaze_calib_online2D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
        run_report.write(cg_on2d_s + '\n')
        cg_on2d_d = [cg_on2d_d[0], 'Calib', 'Online2D', 'Run' + run, cg_on2d_d[1], cg_on2d_d[2]]
        gaze_report = gaze_report.append(pd.Series(cg_on2d_d, index=gaze_report.columns), ignore_index=True)


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

        run_eye_file = File_Source(g_pool, source_path=cfg['run' + run + '_run_mp4'])
        # overwrite dummy variable set when loading intrinsics file...
        #run_eye_file._intrinsics.focal_length = cfg['focal_length']
        #g_pool.capture.intrinsics.focal_length = cfg['focal_length']
        if cfg['apply_qc']:
            run_t_stamps = run_eye_file.timestamps
            diff_list, gap_idx = assess_timegaps(run_t_stamps, cfg['time_threshold'])
            if len(gap_idx) > 0:
                #export as .tsv
                np.savetxt(os.path.join(cfg['out_dir'], 'qc', 'run_framegaps_run' + run + '.tsv'), np.array(gap_idx), delimiter="\t")

        if cfg['use3D']:

            if cfg['freeze_3d_at_test']:
                detect_3d.detector._long_term_schedule.pause()
                detect_3d.detector._ult_long_term_schedule.pause()

        # Predict run's pupils
        run_pupils_2d, run_pupils_3d = predict_all_those_pupils(cfg, detect_2d, run_eye_file, detect_3d, 'run' + run + '_data')

        # QC the offline pupils
        if cfg['apply_qc']:
            rp_off2d_s, rp_off2d_d = qc_report(run_pupils_2d, cfg['out_dir'] + '/qc', 'pupil_run_offline2D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
            run_report.write(rp_off2d_s + '\n')
            rp_off2d_d = [rp_off2d_d[0], 'Run', 'Offline2D', 'Run' + run, rp_off2d_d[1]]
            pupil_report = pupil_report.append(pd.Series(rp_off2d_d, index=pupil_report.columns), ignore_index=True)
            if cfg['use3D']:
                rp_off3d_s, rp_off3d_d = qc_report(run_pupils_3d, cfg['out_dir'] + '/qc', 'pupil_run_offline3D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
                run_report.write(rp_off3d_s + '\n')
                rp_off3d_d = [rp_off3d_d[0], 'Run', 'Offline3D', 'Run' + run, rp_off3d_d[1]]
                pupil_report = pupil_report.append(pd.Series(rp_off3d_d, index=pupil_report.columns), ignore_index=True)

    # QC the online pupils
    if cfg['apply_qc']:
        run_online_pupils = load_pldata_file(cfg['run' + run + '_run_mp4'][:-9], 'pupil')[0]
        rp_on2d_s, rp_on2d_d = qc_report(run_online_pupils, cfg['out_dir'] + '/qc', 'pupil_run_online2D_run' + run, 'pupils', cfg['pupil_confidence_threshold'])
        run_report.write(rp_on2d_s + '\n')
        rp_on2d_d = [rp_on2d_d[0], 'Run', 'Online2D', 'Run' + run, rp_on2d_d[1]]
        pupil_report = pupil_report.append(pd.Series(rp_on2d_d, index=pupil_report.columns), ignore_index=True)

    # load run's online pupils
    else:
        seri_run_pupils_2d = load_pldata_file(cfg['run' + run + '_previous_run_2dpupils'], cfg['run' + run + '_prev_run_pup2D_name'])
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
            seri_run_pupils_3d = load_pldata_file(cfg['run' + run + '_previous_run_3dpupils'], cfg['run' + run + '_prev_run_pup3D_name'])
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
    gaze_data_2d, gaze_data_3d = map_all_those_pupils_to_gaze(cfg, gazemap_2d, gazemap_3d, run_pupils_2d, run_pupils_3d, 'run' + run + '_rundata')

    if cfg['apply_qc']:
        rg_off2d_s, rg_off2d_d = qc_report(gaze_data_2d, cfg['out_dir'] + '/qc', 'gaze_run_offline2D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
        run_report.write(rg_off2d_s + '\n')
        rg_off2d_d = [rg_off2d_d[0], 'Run', 'Offline2D', 'Run' + run, rg_off2d_d[1], rg_off2d_d[2]]
        gaze_report = gaze_report.append(pd.Series(rg_off2d_d, index=gaze_report.columns), ignore_index=True)
        if cfg['use3D']:
            rg_off3d_s, rg_off3d_d = qc_report(gaze_data_3d, cfg['out_dir'] + '/qc', 'gaze_run_offline3D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
            run_report.write(rg_off3d_s + '\n')
            rg_off3d_d = [rg_off3d_d[0], 'Run', 'Offline3D', 'Run' + run, rg_off3d_d[1], rg_off3d_d[2]]
            gaze_report = gaze_report.append(pd.Series(rg_off3d_d, index=gaze_report.columns), ignore_index=True)
    # QC online gaze
    if cfg['apply_qc']:
        run_online_gaze = load_pldata_file(cfg['run' + run + '_run_mp4'][:-9], 'gaze')[0]
        rg_on2d_s, rg_on2d_d = qc_report(run_online_gaze, cfg['out_dir'] + '/qc', 'gaze_run_online2D_run' + run, 'gaze', cfg['gaze_confidence_threshold'])
        run_report.write(rg_on2d_s + '\n')
        rg_on2d_d = [rg_on2d_d[0], 'Run', 'Online2D', 'Run' + run, rg_on2d_d[1], rg_on2d_d[2]]
        gaze_report = gaze_report.append(pd.Series(rg_on2d_d, index=gaze_report.columns), ignore_index=True)

    run_report.close()
    return pupil_report, gaze_report


if __name__ == '__main__':
    '''
    Script outputs offline pupil and gaze outputs, and (optionally) performs quality checks on them,
    based on functions from quality_check_THINGS.py (which can be ran only to QC online pupils and gaze)

    Both scripts use the same config file, for convenience
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

    pct = str(cfg['pupil_confidence_threshold'])
    gct = str(cfg['gaze_confidence_threshold'])

    pupil_reports = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + pct + ' Confidence Threshold'])
    gaze_reports = pd.DataFrame(columns=['Name', 'Type', 'Processing', 'Run', 'Below ' + gct + ' Confidence Threshold', 'Outside Screen Area'])

    for run in cfg['runs']:
        print('Run ' + str(run))
        try:
            pupil_report, gaze_report = map_run_gaze(cfg, run)
            pupil_reports = pd.concat((pupil_reports, pupil_report), ignore_index=True)
            gaze_reports = pd.concat((gaze_reports, gaze_report), ignore_index=True)
        except:
            print('Something went wrong processing run ' + run)

    pupil_reports.to_csv(cfg['out_dir'] +'/qc/' + cfg['subject'] + '_ses' + cfg['session'] + '_pupil_report.tsv', sep='\t', header=True, index=False)
    gaze_reports.to_csv(cfg['out_dir'] +'/qc/' + cfg['subject'] + '_ses' + cfg['session'] + '_gaze_report.tsv', sep='\t', header=True, index=False)
