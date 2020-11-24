import os, sys
import bids
import bioread
import pandas
import pathlib
import argparse
import logging
import datetime
from pytz import timezone


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="match acq_files to bold MRI series",
    )
    parser.add_argument("bids_path", type=pathlib.Path, help="BIDS folder")
    parser.add_argument(
        "biopac_data_path",
        type=pathlib.Path,
        help="folder where biopac data were dumped",
    )
    parser.add_argument(
        "--debug",
        dest="debug_level",
        action="store",
        default="info",
        help="debug level",
    )
    return parser.parse_args()


def match_all_bolds(bids_path, biopac_path):

    tz = timezone("Canada/Eastern")

    # get acqk file starts and end datetimes<
    acqk_files = sorted(list(biopac_path.glob("*.acq")))
    acqk_files_startends = []
    for acqk in acqk_files:
        try:
            acq_h = bioread.read_headers(str(acqk))
            acq_start = acq_h.earliest_marker_created_at
            if acq_start is None:
                logging.error(f"no start marker in: {acqk}")
                continue
            acq_end = acq_start
            if len(acq_h.time_index):
                acq_end = acq_start + datetime.timedelta(seconds=acq_h.time_index[-1])
            acqk_files_startends.append((acqk, acq_start, acq_end))
        except Exception as e:
            logging.error(f"read error for file: {acqk}")

    sourcedata = bids_path / "sourcedata" / "physio"
    sourcedata.mkdir(parents=True, exist_ok=True)
    sessions_list = sorted(bids_path.glob("sub-*/ses-*"))
    for session in sessions_list:
        session_bids_path = session.relative_to(bids_path)
        session_sourcedata = sourcedata / session_bids_path
        session_sourcedata.mkdir(parents=True, exist_ok=True)
        sub_ses_prefix = str(session_bids_path).replace(os.path.sep, "_")
        scans = pandas.read_csv(
            session / (sub_ses_prefix + "_scans.tsv"),
            delimiter="\t",
            parse_dates=["acq_time"],
        )
        list_matches_out = session_sourcedata / (
            sub_ses_prefix + "_physio_fmri_matches.tsv"
        )
        if list_matches_out.exists():
            continue
        matches = []
        for idx, scan in scans.iterrows():
            if "_bold.nii.gz" in scan.filename:
                acq_time = tz.localize(scan.acq_time.to_pydatetime())
                acq_files = [
                    acqk_wtiming
                    for acqk_wtiming in acqk_files_startends
                    if acqk_wtiming[1] < acq_time and acqk_wtiming[2] > acq_time
                ]
                if len(acq_files) == 0:
                    logging.error(f"No acq file found for: {scan.filename}")
                    matches.append((session / scan.filename, None))
                else:
                    if len(acq_files) > 1:
                        if not all(
                            [acq[1] == acq_files[0][1] for acq in acq_files[1:]]
                        ):  # duplicated files
                            logging.warning(
                                f"More that one acq file found for: {scan.filename} \n {acq_files}"
                            )
                    bname = os.path.basename(acq_files[0][0])
                    dest_path = session_sourcedata / bname
                    matches.append(
                        (session / scan.filename, dest_path.relative_to(bids_path))
                    )
                    if not dest_path.exists() and acq_files[0][0].exists():
                        logging.info(f"moving {acq_files[0][0]} to {dest_path}")
                        os.rename(acq_files[0][0], dest_path)
        list_matches_out.write_text("\n".join([f"{m[0]}\t{m[1]}" for m in matches]))


if __name__ == "__main__":
    parsed = parse_args()
    logging.basicConfig(level=logging.getLevelName(parsed.debug_level.upper()))
    match_all_bolds(parsed.bids_path, parsed.biopac_data_path)
