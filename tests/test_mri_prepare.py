import os
import json
import pandas as pd
from datetime import datetime, timedelta
from mri.prepare.anonymize_dates import replace_dates


class FakeBIDSObject(object):
    pass


def test_replace_dates(tmp_path):
    first_date = "2022-06-01T00:00:00.000000"
    first_date = datetime.fromisoformat(first_date)
    rel_times = [1.13845, 12.33666789, 456.2]
    acq_times = [datetime.isoformat(first_date + t * timedelta(days=1)) for t in rel_times]
    scan_dict = {"acq_time": acq_times}
    df = pd.DataFrame.from_dict(scan_dict)
    scan_path = os.path.join(tmp_path, "scan.tsv")
    df.to_csv(scan_path, sep="\t")
    sidecar_dict = {
        "acq_time": {
            "LongName": "Acquisition time",
            "Description": "Acquisition time of the particular scan",
        }
    }
    sidecar_path = os.path.join(tmp_path, "scan.json")
    with open(sidecar_path, "w") as sidecar_file:
        json.dump(sidecar_dict, sidecar_file, indent=2)

    scan_obj = FakeBIDSObject()
    scan_obj.path = scan_path
    modified_files = replace_dates([scan_obj], first_date)
    assert len(modified_files) == 1 and modified_files[0] == scan_path, print(
        "tsv file not reported as modified."
    )

    new_df = pd.read_csv(scan_path, sep="\t")
    assert "acq_time" not in new_df.columns, "'acq_time' column still in tsv file."
    assert "rel_acq_time" in new_df.columns, "'rel_acq_time' column not in tsv file."
    for i in range(len(rel_times)):
        assert new_df["rel_acq_time"][i] == rel_times[i], "Relative time not correct."

    with open(sidecar_path, "r") as sidecar_file:
        new_sidecar = json.load(sidecar_file)

    assert (
        "acq_time" not in new_sidecar and "rel_acq_time" in new_sidecar
    ), "Sidecar not properly updated."
