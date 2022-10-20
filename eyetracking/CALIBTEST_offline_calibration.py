import os, sys, platform, json, glob
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from types import SimpleNamespace

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

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


def get_file_list(cfg):
    data_path = cfg['data_dir']
    calib_list = sorted(glob.glob(os.path.join(data_path, '*_calib-data.npz')))

    if cfg['validation']:
        '''
        Only consider runs that have both calibration and validation data
        '''
        c_list = []
        v_list = []
        for c in calib_list:
            sub, ses, file_num, task, suffix = os.path.basename(c).split('_')
            v_name = '*' + file_num + '*' + task.split('-')[-1] + '*valid-data.npz'
            v = glob.glob(os.path.join(data_path, v_name))
            if len(v) == 1:
                v_list.append(v[0])
                c_list.append(c)

        return c_list, v_list

    return calib_list, []


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


def get_marker_dictionary(ref_list):
    position_list = []
    markers_dict = {}
    count = 0

    for i in range(len(ref_list)):
        m = ref_list[i]
        if not (m['norm_pos']) in position_list:
            markers_dict[count] = {
                'norm_pos': m['norm_pos'],
                'screen_pos': m['screen_pos'],
                'onset': m['timestamp'],
                'offset': -1.0,
            }
            count += 1
            position_list.append(m['norm_pos'])
        elif m['timestamp'] > markers_dict[count-1]['offset']:
            markers_dict[count-1]['offset'] = m['timestamp']

    return markers_dict


def assign_gaze_to_markers(gaze_list, markers_dict):
    '''
    Assign gaze to markers based on their onset.
    A gaze is assigned to a marker if it is captured from 500ms to  its ONSET overlaps with the time the marker is on the screen
    '''
    i = 0
    #print(markers_dict[0]['onset'], fixation_list[0]['timestamp'])
    for count in range(len(markers_dict.keys())):
        marker = markers_dict[count]
        gaze_data = {'timestamps': [],
                     'norm_pos': [],
                     'confidence': [],
                    }

        while i < len(gaze_list) and gaze_list[i]['timestamp'] < marker['onset']:
            i += 1

        while i < len(gaze_list) and gaze_list[i]['timestamp'] < marker['offset']:
            gaze = gaze_list[i]

            gaze_data['timestamps'].append(gaze['timestamp'])
            gaze_data['norm_pos'].append(gaze['norm_pos'])
            gaze_data['confidence'].append(gaze['confidence'])

            i += 1

        markers_dict[count]['gaze_data'] = gaze_data

    return markers_dict


def gaze_to_marker_distances(markers_dict, conf_thresh = 0.9):
    # distance between eye and screen, in pixels (estimated w pytagore + screen w and h in deg of vis angle)
    dist_in_pix = 4164 # in pixels

    val_qc = []

    for count in range(len(markers_dict.keys())):
        m = markers_dict[count]
        print('Marker ' + str(count) + ', Normalized position: ' +  str(m['norm_pos']))
        # transform marker's normalized position into dim = (3,) vector in pixel space
        m_vecpos = np.concatenate(((np.array(m['norm_pos']) - 0.5)*(1280, 1024), np.array([dist_in_pix])), axis=0)

        if len(m['gaze_data']['timestamps']) > 0:
            # filtrate gaze based on confidence threshold
            g_conf = np.array(m['gaze_data']['confidence'])
            g_filter = g_conf > conf_thresh

            g_pos = np.array(m['gaze_data']['norm_pos'])[g_filter]
            g_times = np.array(m['gaze_data']['timestamps'])[g_filter]

            gaze = (g_pos - 0.5)*(1280, 1024)
            gaze_vecpos = np.concatenate((gaze, np.repeat(dist_in_pix, len(gaze)).reshape((-1, 1))), axis=1)

            distances = []
            for gz_vec in gaze_vecpos:
                vectors = np.stack((m_vecpos, gz_vec), axis=0)
                distance = np.rad2deg(np.arccos(1.0 - pdist(vectors, metric='cosine')))
                distances.append(distance[0])

            distances = np.array(distances)
            assert(len(distances)==len(g_times))
            markers_dict[count]['gaze_data']['distances'] = {'distances': distances,
                                                             'timestamps': g_times,
                                                             }

            num_gz = len(distances)
            good = np.sum(distances < 0.5) / num_gz
            fair = np.sum((distances >= 0.5)*(distances < 1.5)) / num_gz
            poor = np.sum(distances >= 1.5) / num_gz

            print('Total gaze:' + str(num_gz) + ' , Good:' + str(good) + ' , Fair:' + str(fair) + ' , Poor:' + str(poor))
            val_qc.append({
                'marker': count,
                'norm_pos': m['norm_pos'],
                'num_gz': num_gz,
                'good': good,
                'fair': fair,
                'poor': poor
            })
        else:
            val_qc.append({
                'marker': count,
                'norm_pos': m['norm_pos'],
                'num_gz': 0
            })

    return markers_dict, val_qc


def export_figs(markers, gaze, out_dir, label, conf_thresh):

    markers_dict = get_marker_dictionary(markers)
    markers_dict = assign_gaze_to_markers(gaze, markers_dict)

    markers_x = []
    markers_y = []

    gaze_x = []
    gaze_y = []

    for k in markers_dict.keys():
        m_pos = markers_dict[k]['norm_pos']
        x = int(m_pos[0] * 1280)
        y = int(m_pos[1] * 1024)

        markers_x.append(x)
        markers_y.append(y)

        gaze_dict = markers_dict[k]['gaze_data']

        for i in range(len(gaze_dict['confidence'])):
            if gaze_dict['confidence'][i] > conf_thresh:
                g_pos = gaze_dict['norm_pos'][i]
                g_x = int(g_pos[0] * 1280)
                g_y = int(g_pos[1] * 1024)
                gaze_x.append(g_x)
                gaze_y.append(g_y)

    plt.ylim(0, 1024)
    plt.xlim(0, 1280)
    plt.scatter(gaze_x, gaze_y, c='xkcd:sky blue', s=1, alpha=0.2)
    plt.scatter(markers_x, markers_y, c='xkcd:red', s=25, alpha=1)
    plt.savefig(os.path.join(out_dir, label + '_dist_in_pix.png'))
    plt.clf()

    markers_dict, val_qc = gaze_to_marker_distances(markers_dict, conf_thresh)
    for j in range(len(markers_dict.keys())):
        d = markers_dict[j]['gaze_data']['distances']['distances']
        t = np.array(markers_dict[j]['gaze_data']['distances']['timestamps'])
        plt.plot(t, d)
    plt.ylim(0, 16)
    plt.xlabel('time (s)')
    plt.ylabel('absolute distance (deg vis angle)')
    plt.savefig(os.path.join(out_dir, label + '_dist_in_deg.png'))
    plt.clf()

def calibrate_offline(cfg, calib_path, valid_path=None):
    '''
    Step 0. Set up
    '''
    # Get paths for calibration (and validation) eye video(s)
    sub, ses, file_num, ctask, suffix = os.path.basename(calib_path).split('_')
    p_name = sub + '_' + ses + '_' + file_num + '.pupil'
    calib_mp4 = os.path.join(os.path.dirname(calib_path), p_name, ctask, '000', 'eye0.mp4')
    cal_out_name = sub + '_' + ses + '_' + file_num + '_' + ctask

    if cfg['validation']:
        sub, ses, file_num, vtask, suffix = os.path.basename(valid_path).split('_')
        valib_mp4 = os.path.join(os.path.dirname(valid_path), p_name, vtask, '000', 'eye0.mp4')
        val_out_name = sub + '_' + ses + '_' + file_num + '_' + vtask

    detect_2d = None
    detect_3d = None

    gazemap_2d = None
    gazemap_3d = None

    val_gazemap_2d = None
    val_gazemap_3d = None

    g_pool = make_detection_gpool()

    if cfg['overwrite_camera_intrinsics']:
        '''
        TODO: generate new eye0.instrinsics file to replace the existing one...
        Essential to use 3D pupil detector
        make_intrinsics(cfg['calib_mp4'][:-4]+'.instrinsics')
        '''
        pass

    '''
    Step 1. Detect calibration pupils offline with 2D (and 3D pupil) detector(s)
    '''
    # Instantiate pupil detectors
    calib_eye_file = File_Source(g_pool, source_path=calib_mp4)

    detect_2d = Detector2DPlugin(g_pool, cfg['properties'])
    if cfg['use3D']:
        detect_3d = Pye3DPlugin(g_pool)

    # Predict calibration pupils offline
    calib_pupils_2d, calib_pupils_3d = predict_all_those_pupils(cfg, detect_2d, calib_eye_file, detect_3d, cal_out_name)

    '''
    Step 2. Initialize the 2d (and 3d) Gazer model(s)
    '''
    #g_pool.capture.intrinsics.focal_length = cfg['focal_length']
    g_pool.min_calibration_confidence = cfg['min_calibration_confidence']
    g_pool.ipc_pub = FakeIPC()
    g_pool.get_timestamp = time

    # Use marker and pupil data to fit a gazer model
    cal_file = np.load(calib_path, allow_pickle=True)

    cal_data_2d = {}
    cal_data_2d['ref_list'] = cal_file['markers'].tolist()
    # use  offline pupils detected in step 1 above rather than the online pupils saved w the markers
    cal_data_2d['pupil_list'] = calib_pupils_2d

    # initialization calls Gazer_base's method fit_on_calib_data() to fit calibration parameters from markers and pupils
    gazemap_2d = Gazer2D(g_pool, calib_data=cal_data_2d)
    # for now: separate gazers to map val and calib data, to play it safe (more important for 3d algo though)
    if cfg['validation']:
        val_gazemap_2d = Gazer2D(g_pool, calib_data=cal_data_2d)

    # sanity check
    assert(gazemap_2d.right_model._is_fitted == True)
    assert(gazemap_2d.binocular_model._is_fitted == False)
    assert(gazemap_2d.left_model._is_fitted == False)

    # export newly created calibration parameters as .plcal file
    cal_params = {}
    cal_params['data'] = {}
    cal_params['data']['calib_params'] = gazemap_2d.get_params()
    save_object(cal_params, os.path.join(cfg['out_dir'], cal_out_name + '_offlinecalib_2D.plcal'))


    if cfg['use3D']:
        cal_data_3d = {}
        cal_data_3d['ref_list'] = cal_file['markers'].tolist()
        cal_data_3d['pupil_list'] = calib_pupils_3d

        # separate gazers to map calibration and validation gaze
        gazemap_3d = Gazer3D(g_pool, calib_data=cal_data_3d)
        val_gazemap_3d = Gazer3D(g_pool, calib_data=cal_data_3d)

        assert(gazemap_3d.right_model._is_fitted == True)
        assert(gazemap_3d.binocular_model._is_fitted == False)
        assert(gazemap_3d.left_model._is_fitted == False)

        # export newly created calibration parameters as .plcal file
        cal_params = {}
        cal_params['data'] = {}
        cal_params['data']['calib_params'] = gazemap_3d.get_params()
        save_object(cal_params, os.path.join(cfg['out_dir'], cal_out_name + '_offlinecalib_3D.plcal'))

    # map calibration pupils to gaze
    calib_gaze_2d, calib_gaze_3d = map_all_those_pupils_to_gaze(cfg, gazemap_2d, gazemap_3d, calib_pupils_2d, calib_pupils_3d, cal_out_name)


    '''
    Step 3. (optional) predict validation pupils and map those pupils to gaze position
    '''
    if cfg['validation']:
        # predict validation pupils offline
        if cfg['overwrite_camera_intrinsics']:
            '''
            TODO: generate new eye0.instrinsics file...
            make_intrinsics(cfg['run_mp4'][:-4]+'.instrinsics')
            '''
            pass

        valid_eye_file = File_Source(g_pool, source_path=valib_mp4)
        # overwrite dummy variable set when loading intrinsics file...
        #run_eye_file._intrinsics.focal_length = cfg['focal_length']
        #g_pool.capture.intrinsics.focal_length = cfg['focal_length']

        if cfg['use3D']:
            if cfg['freeze_3d_at_test']:
                detect_3d.detector._long_term_schedule.pause()
                detect_3d.detector._ult_long_term_schedule.pause()

        # Predict validation pupils
        valid_pupils_2d, valid_pupils_3d = predict_all_those_pupils(cfg, detect_2d, valid_eye_file, detect_3d, val_out_name)

        # map validation pupils to gaze
        valid_gaze_2d, valid_gaze_3d = map_all_those_pupils_to_gaze(cfg, val_gazemap_2d, val_gazemap_3d, valid_pupils_2d, valid_pupils_3d, val_out_name)

    '''
    Step 4. export figures to visualize fit
    '''
    export_figs(cal_file['markers'], calib_gaze_2d, cfg['out_dir'], cal_out_name+'_2d', conf_thresh=0.85)

    if cfg['validation']:
        val_file = np.load(valid_path, allow_pickle=True)
        export_figs(val_file['markers'], valid_gaze_2d,  cfg['out_dir'], val_out_name+'_2d', conf_thresh=0.85)


if __name__ == '__main__':
    '''
    Script uses eye video captured during calibration,
    re-computes the calibration offline and export the gaze mapping, for QC purposes

    Options include using pupil's 3D pupil detection model (WIP... need params),
    and re-mapping the gaze of an (optional) validation sequence based on the offline calibration

    Options are specified in a config file (.json)
    e.g., ./config/config_calibtest/config_calibtest_sub03.json
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

    calib_files, valid_files = get_file_list(cfg)

    for i in range(len(calib_files)):
        calib_file = calib_files[i]
        valid_file = valid_files[i] if cfg['validation'] else None

        calibrate_offline(cfg, calib_file, valid_file)
