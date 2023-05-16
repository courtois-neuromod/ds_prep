import os, glob
import pandas as pd
import numpy as np


# List CNeuromod1 datasets that include eyetracking data
ds_specs = {
    'emotionsvideos': {},
    'floc': {},
    'friends': {},
    'friends_fix': {},
    'harrypotter': {},
    'mario': {},
    'mario3': {},
    'mariostars': {},
    'movie10fix': {},
    'narratives': {},
    'petitprince': {},
    'retino': {},
    'shinobi': {},
    'things': {},
    'triplets': {}
}

def get_arguments():

    parser = argparse.ArgumentParser(description='clean up, label, QC and bids-formats the triplets eye tracking dataset')
    parser.add_argument('--in_path', type=str, required=True, help='absolute path to directory that contains all data')
    parser.add_argument('--out_path', type=str, default='./test.tsv', help='absolute path to output file')
    args = parser.parse_args()

    return args


def compile_file_list(in_path):

    col_names = ['subject', 'session', 'run', 'task', 'file_number', 'complete_files', 'has_log']
    df_files = pd.DataFrame(columns=col_names)

    # on elm, for triplets : in_path = '/unf/eyetracker/neuromod/triplets/sourcedata'
    ses_list = sorted(glob.glob(f'{in_path}/sub-*/ses-*'))

    for ses in ses_list:
        [sub_num, ses_num] = ses.split('/')[-2:]
        events_list = sorted(glob.glob(f'{ses}/*task*events.tsv'))
        for event in events_list:
            ev_file = os.path.basename(event)
            [sub, ses, fnum, task_type, run_num, appendix] = ev_file.split('_')
            assert sub == sub_num
            assert ses_num == ses

            has_log = glob.glob(f'{ses}/{sub_num}_{ses_num}_{fnum}.log')

            pupil_path = f'{ses}/{sub_num}_{ses_num}_{fnum}.pupil'
            has_pupil = glob.glob(f'{pupil_path}/{task_type}_{run_num}/000/pupil.pldata')
            has_eyemv = glob.glob(f'{pupil_path}/{task_type}_{run_num}/000/eye0.mp4')
            has_gaze = glob.glob(f'{pupil_path}/{task_type}_{run_num}/000/gaze.pldata')

            run_data = [sub_num, ses_num, run_num, task_type, fnum, (has_pupil+has_eyemv+has_gaze)==3, has_log]
            df_files = df_files.append(pd.Series(run_data, index=df_files.columns), ignore_index=True)

    return df_files


def main():

    args = get_arguments()

    in_path = args.in_path
    # e.g., (elm): /unf/eyetracker/neuromod/triplets/sourcedata
    out_file = args.out_path

    '''
    Step 1: compile overview of available files
    export as text file?
    '''
    file_report = compile_file_list(in_path)
    file_report.to_csv(f'{out_path}/file_list.tsv', sep='\t', header=True, index=False)

    '''
    Step 2: export gaze and pupil files from pupil dataset to numpy .npz format
    For each run, plot the raw gaze / pupil data and export chart(s) for QCing
    '''
    

    '''
    Step 3: manual QCing... rate quality of each run, log in spreadsheet
    '''

    '''
    Step 4: apply drift correction to gaze (if possible...) based on known fixations

    Triplets task details here:
    https://github.com/courtois-neuromod/task_stimuli/blob/4d1e66bdb66b722eb25a886a0008e2668054e470/src/tasks/language.py#L115

    Fixation in the center of the screen: (0,0)
    Task begins with fixation until the first trial's onset (~6s from task onset)
    For each trial, the 3 words (triplets) appear from the trial's onset until onset + duration (4s);
    Then, a central fixation point is shown from the time of [onset + duration] until [onset + duration + ISI (varied) - 0.1s]
    '''

    '''
    Step 5: export to tsv.gz format following bids extension guidelines
    - pupil size and confidence
    - gaze position (before and after correction) and confidence;
    - Add gaze position in different metrics? in pixels (on screen), and then in pixels (stimulus image), and then normalized screen
    - set timestamp to 0 = task onset, export in ms (integer)

    '''


if __name__ == '__main__':
    sys.exit(main())
