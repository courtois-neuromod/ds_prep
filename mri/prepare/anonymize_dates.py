import os
import json
import logging
import argparse
import datalad.api
import pandas as pd
from bids import BIDSLayout
from datetime import datetime, timedelta
from datalad.support.annexrepo import AnnexRepo

PYBIDS_CACHE_PATH = ".pybids_cache"

def update_sidecar(sidecar_path):
    with open(sidecar_path, "r") as sidecar_file:
        sidecar = json.load(sidecar_file)
    sidecar["rel_acq_time"] = {
        "LongName": "Relative acquisition time",
        "Description": "Acquisition time of the particular scan, "
        "relative to the first scan of the first session, in days.",
    }
    del sidecar["acq_time"]
    with open(sidecar_path, "w") as sidecar_file:
        json.dump(sidecar, sidecar_file, indent=2)


def replace_dates(scans_list, first_date):
    """Replace dates in scans files with relative dates from the 1st session."""
    modified_files = []
    for scans in scans_list:
        scans_df = pd.read_csv(scans.path, sep="\t")
        if "acq_time" in scans_df.columns:
            scans_df["rel_acq_time"] = scans_df.apply(
                lambda row: (datetime.fromisoformat(row["acq_time"]) - first_date)
                / timedelta(days=1),
                axis=1,
            )
            scans_df = scans_df.drop(columns=["acq_time"])
            scans_df.to_csv(scans.path, sep="\t")
            modified_files.append(scans.path)
            sidecar_path = os.path.splitext(scans.path)[0] + ".json"
            if os.path.exists(sidecar_path):
                update_sidecar(sidecar_path)
    return modified_files


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Replace acquisition time in scans.tsv files by the relative time of acquisition, "
        "relative to the first session for each subject, in days.",
    )

    parser.add_argument("bids_path", help="BIDS folder in which to update the dates.")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force pyBIDS reset_database and reindexing",
    )
    parser.add_argument(
        "--datalad",
        action="store_true",
        help="Update distribution-restrictions metadata and commit changes",
    )
    parser.add_argument(
        "--debug",
        dest="debug_level",
        action="store",
        default="info",
        help="debug level",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    logging.basicConfig(level=logging.getLevelName(args.debug_level.upper()))

    bids_path = os.path.abspath(args.bids_path)
    pybids_cache_path = os.path.join(bids_path, PYBIDS_CACHE_PATH)

    layout = BIDSLayout(
        bids_path,
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        validate=False,
    )

    if args.datalad:
        annex_repo = AnnexRepo(args.bids_path)

    subject_list = layout.get_subjects()
    modified_files = []

    for subject in subject_list:
        first_scan = layout.get(subject=subject, suffix="scans", extension=".tsv", session="001")[0]
        first_date = pd.read_csv(first_scan.path, sep="\t")["acq_time"][0]
        first_date = datetime.fromisoformat(first_date)
        scans_list = layout.get(subject=subject, suffix="scans", extension=".tsv")

        if args.datalad:
            # get scans files
            for scans in scans_list:
                datalad.api.get(scans.path)
            # unlock before making any change to avoid unwanted save
            annex_repo.unlock([scans.path for scans in scans_list])

        modified_files += replace_dates(scans_list, first_date)

    root_sidecar = os.path.join(bids_path,'scans.json')
    if os.path.exists(root_sidecar):
        update_sidecar(root_sidecar)
        modified_files.append(root_sidecar)
        
    if args.datalad and len(modified_files):
        logging.info("saving files and metadata changes in datalad")
        annex_repo.set_metadata(modified_files, remove={"distribution-restrictions": "sensitive"})
        datalad.api.save(
            modified_files,
            message="make accquisition time relative to the first session and update distribution-restrictions",
        )


if __name__ == "__main__":
    main()
 
