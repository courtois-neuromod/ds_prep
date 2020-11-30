import sys, os
import shutil, stat
from bids import BIDSLayout
import json
import logging
import argparse
import numpy as np
import datetime
from operator import itemgetter
from heudiconv.utils import json_dumps_pretty

PYBIDS_CACHE_PATH = ".pybids_cache"


def fill_intended_for(args):
    path = os.path.abspath(args.bids_path)
    pybids_cache_path = os.path.join(path, PYBIDS_CACHE_PATH)

    layout = BIDSLayout(
        path,
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        validate=False,
    )
    extra_filters = {}
    if args.participant_label:
        extra_filters["subject"] = args.participant_label
    if args.session_label:
        extra_filters["session"] = args.session_label
    bolds = layout.get(suffix="bold", extension=".nii.gz", **extra_filters)
    json_to_modify = dict()

    bolds_with_no_fmap = []
    bolds_with_shim_mismatch = []

    for bold in bolds:
        bold_scan_time = datetime.datetime.strptime(
            bold.tags["AcquisitionTime"].value, "%H:%M:%S.%f"
        )

        fmaps = layout.get(
            suffix="epi",
            extension=".nii.gz",
            acquisition="sbref",
            subject=bold.entities["subject"],
            session=bold.entities["session"],
        )

        shim_settings = bold.tags["ShimSetting"].value
        # First: get epi fieldmaps with similar ShimSetting
        fmaps_match = [
            fm
            for fm in fmaps
            if np.allclose(fm.tags["ShimSetting"].value, shim_settings)
        ]
        pedirs = set([fm.tags["PhaseEncodingDirection"].value for fm in fmaps_match])

        # Second: if not 2 fmap found we extend our search to similar ImageOrientationPatient
        if len(fmaps_match) < 2 or len(pedirs) < 2:
            bolds_with_shim_mismatch.append(bold)
            logging.warning(
                f"We couldn't find two epi fieldmaps with matching ShimSettings and two pedirs for: {bold.path}. "
                "Including other based on ImageOrientationPatient."
            )
            fmaps_match.extend(
                [
                    fm
                    for fm in fmaps
                    if np.allclose(
                        fm.tags["ImageOrientationPatientDICOM"].value,
                        bold.tags["ImageOrientationPatientDICOM"].value,
                    )
                    and fm not in fmaps_match
                ]
            )

            pedirs = set(
                [fm.tags["PhaseEncodingDirection"].value for fm in fmaps_match]
            )

        # get all fmap possible
        if len(fmaps_match) < 2 or len(pedirs) < 2:
            logging.error(
                f"We couldn't find two epi fieldmaps with matching ImageOrientationPatient and two pedirs for {bold.path}. "
                "Please review manually."
            )
            bolds_with_no_fmap.append(bold)
            continue
            # TODO: maybe match on time distance
            # fmaps_match = fmaps

        # only get 2 images with opposed pedir
        fmaps_match_pe_pos = [
            fm
            for fm in fmaps_match
            if "-" not in fm.tags["PhaseEncodingDirection"].value
        ]
        fmaps_match_pe_neg = [
            fm for fm in fmaps_match if "-" in fm.tags["PhaseEncodingDirection"].value
        ]

        if not fmaps_match_pe_pos or not fmaps_match_pe_neg:
            logging.error("no matching fieldmaps")
            continue

        for fmap in [fmaps_match_pe_pos, fmaps_match_pe_neg]:
            fmaps_time_diffs = sorted(
                [
                    (
                        fm,
                        datetime.datetime.strptime(
                            fm.tags.get("AcquisitionTime").value, "%H:%M:%S.%f"
                        )
                        - bold_scan_time,
                    )
                    for fm in fmap
                ],
                key=itemgetter(1),
            )

            if args.match_strategy == "before":
                delta_for_sbref = datetime.timedelta(seconds=0)
                match_fmap = [
                    fm for fm, ftd in fmaps_time_diffs if ftd <= delta_for_sbref
                ]
                if len(match_fmap):
                    match_fmap = match_fmap[-1]
                else:
                    logging.warning(
                        f"No fmap matched the {args.match_strategy} strategy for {bold.path}, taking the first match after scan."
                    )
                    match_fmap = fmaps_time_diffs[0][0]
            elif args.match_strategy == "after":
                # to match the sbref with the corresponding bold, there is a diff of ~11sec
                delta_for_sbref = datetime.timedelta(seconds=-15)
                match_fmap = [
                    fm for fm, ftd in fmaps_time_diffs if ftd >= delta_for_sbref
                ]
                if len(match_fmap):
                    match_fmap = match_fmap[0]
                else:
                    logging.warning(
                        f"No fmap matched the {args.match_strategy} strategy for {bold.path}, taking the first match before scan."
                    )
                    match_fmap = fmaps_time_diffs[-1][0]
            if ("IntendedFor" not in match_fmap.tags) or (
                bold.path not in match_fmap.tags.get("IntendedFor").value
            ):
                fmap_json_path = match_fmap.get_associations()[0].path
                if fmap_json_path not in json_to_modify:
                    json_to_modify[fmap_json_path] = []
                json_to_modify[fmap_json_path].append(
                    os.path.relpath(bold.path, bold.path.split("ses-")[0])
                )

    for json_path, intendedfor in json_to_modify.items():
        # logging.info("updating %s"%json_path)
        json_path = os.path.join(path, json_path)
        with open(json_path, "r", encoding="utf-8") as fd:
            meta = json.load(fd)
        if "IntendedFor" not in meta:
            meta["IntendedFor"] = []
        meta["IntendedFor"] = sorted(intendedfor)
        # meta['IntendedFor'].extend(intendedfor)
        # meta['IntendedFor'] = sorted(list(set(meta['IntendedFor'])))

        # backup_path = json_path + '.bak'
        # if not os.path.exists(backup_path):
        #    shutil.copyfile(json_path, backup_path)

        file_mask = os.stat(json_path)[stat.ST_MODE]
        os.chmod(json_path, file_mask | stat.S_IWUSR)
        with open(json_path, "w", encoding="utf-8") as fd:
            fd.write(json_dumps_pretty(meta))
        os.chmod(json_path, file_mask)

    if len(bolds_with_no_fmap):
        bolds_with_no_fmap_path = [
            os.path.relpath(bold.path, path) for bold in bolds_with_no_fmap
        ]
        logging.error(
            "No phase-reversed fieldmap was found for the following files:\n"
            + "\n".join(bolds_with_no_fmap_path)
        )
        no_fmap_file = os.path.join(path, "bolds_with_no_fmap.log")
        with open(no_fmap_file, "a") as fd:
            fd.write("\n".join(bolds_with_no_fmap_path) + "\n")
        logging.info("This list was exported in {}".format(no_fmap_file))
    if len(bolds_with_shim_mismatch):
        bolds_with_shim_mismatch_paths = [
            os.path.relpath(bold.path, path) for bold in bolds_with_shim_mismatch
        ]
        logging.error(
            "No phase-reversed with matching ShimSettings was found following files:\n"
            + "\n".join(bolds_with_shim_mismatch_paths)
        )
        shim_mismatch_file = os.path.join(path, "bolds_with_shim_mismatch_paths.log")
        with open(shim_mismatch_file, "a") as fd:
            fd.write("\n".join(bolds_with_shim_mismatch_paths) + "\n")
        logging.info("This list was exported in {}".format(shim_mismatch_file))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Fill "IntendedFor" field of fieldmaps jsons according to scanning parameters.',
    )
    parser.add_argument("bids_path", help="BIDS folder to modify")
    parser.add_argument(
        "--participant-label",
        action="store",
        nargs="+",
        help="a space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed)",
    )
    parser.add_argument(
        "--session-label",
        action="store",
        nargs="+",
        help="a space delimited list of session identifiers or a single "
        "identifier (the ses- prefix can be removed)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force pyBIDS reset_database and reindexing",
    )
    parser.add_argument(
        "--match_strategy",
        choices=["before", "after"],
        default="before",
        help='Strategy to resolve multiple matches: "before"/"after" closest matching in time  ',
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    fill_intended_for(args)
