import bids
import pathlib
from extract_hcptrt import convert_event_file

TIME_CHECK_DELTA_TOL = 180 # tolerance for the match of scan_time to eprime file


def main():
    layout = bids.BIDSLayout('./hcptrt')
    non_rest_tasks = [t for t in layout.get_tasks() if t!='restingstate']
    task_bolds = layout.get(suffix='bold', extension='.nii.gz', task=non_rest_tasks)

    eprime_path = layout.bids_root / 'sourcedata' / 'eprime'

    for task_bold in task_bolds:
        scan_time = task_bold.get_metadata()['AcquisitionTime']
        ents = task_bold.entities
        task_file_path = eprime_path / \
            'sub-%s'%ents['subject'] / \
            'ses-%s'%ents['session'] / \
            'p%02d_%s.txt'%(ents['subject'], ents['task'].upper())
        with open(task_file_path) as f:
            line = ''
            while 'SessionStartDateTimeUtc' not in line:
                line = f.read()
        time_utc = line.split(' ')[-1]
        assert...

        layout.convert_event_file()
