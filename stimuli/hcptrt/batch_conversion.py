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
        eprime_path /
        # SessionStartDateTimeUtc
        layout.convert_event_file()
