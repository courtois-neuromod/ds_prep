import sys, os
import shutil, stat
import pathlib
import re, fnmatch
from bids import BIDSLayout, BIDSLayoutIndexer
from bids.layout import Query, BIDSFile
import json
import logging
import argparse
import numpy as np
import datetime
import coloredlogs
from operator import itemgetter
from typing import Optional, List, Dict, Tuple, Any
from heudiconv.utils import json_dumps_pretty


PYBIDS_CACHE_PATH = ".pybids_cache"

coloredlogs.install()


def _load_bidsignore_(bids_root: pathlib.Path) -> Tuple[Any]:
    """Load .bidsignore file from a BIDS dataset, returns list of regexps"""
    bids_ignore_path = bids_root / ".bidsignore"
    if bids_ignore_path.exists():
        import re
        import fnmatch

        bids_ignores = bids_ignore_path.read_text().splitlines()
        return tuple(
            [
                re.compile(fnmatch.translate(bi))
                for bi in bids_ignores
                if len(bi) and bi.strip()[0] != "#"
            ]
        )
    return tuple()


def get_index_key(bids_file: BIDSFile) -> str:
    if "AcquisitionTime" in bids_file.tags:
        return datetime.datetime.strptime(
            bids_file.tags["AcquisitionTime"].value, "%H:%M:%S.%f"
        )
    elif "SeriesNumber" in bids_file.tags:
        return bids_file.tags["SeriesNumber"].value


def fill_b0_meta(
    bids_path: pathlib.Path,
    force_reindex: bool = False,
    match_strategy: str = "before",
    sloppy: bool = False,
    reset: bool = False,
    **bids_entities: Dict[str, Any],
) -> None:
    # Set BIDS metadata to include B0 intent tags for proper distortion correction
    pybids_cache_path = bids_path / PYBIDS_CACHE_PATH

    bli = BIDSLayoutIndexer(
        ignore=_load_bidsignore_(bids_path),
    )

    layout = BIDSLayout(
        bids_path,
        database_path=pybids_cache_path,
        reset_database=force_reindex,
        validate=False,
        indexer=bli,
        derivatives=False,
    )

    if not "datatype" in bids_entities:
        bids_entities["datatype"] = ["func", "dwi"]
    # if there is sbref for the epi to correct
    # then pepolar epi fmaps should have it too
    use_sbrefs = "sbref" in layout.get_suffix(**bids_entities)

    if "suffix" not in bids_entities:
        bids_entities["suffix"] = ["bold", "dwi", "sbref"]
    bids_entities["extension"] = (".nii.gz",)

    logging.info(bids_entities)
    epis = layout.get(part="mag", **bids_entities) + layout.get(
        part=Query.NONE, **bids_entities
    )
    logging.info("found %d epis", len(epis))

    fmaps_append_b0tags = {}
    fmaps_append_intendedfor = {}

    epis_with_no_fmap = []
    epis_with_shim_mismatch = []

    estimator_pairs = {}

    for epi in epis:
        logging.info(f"matching fmap for {epi.relpath}")
        epi_series_id = (
            os.path.basename(epi.path)
            .split(".")[0]
            .split("_echo")[0]
            .split("_part")[0]
            .replace("-", "_")
        )
        epi_b0fieldsource = epi.entities.get("B0FieldSource", None)
        epi_sorting_value = get_index_key(epi)
        sorting_value_is_date = isinstance(epi_sorting_value, datetime.datetime)

        if epi_b0fieldsource and epi_series_id in epi_b0fieldsource and not reset:
            # that series was already assigned a fieldmap
            continue
        epi_pedir = epi.entities["PhaseEncodingDirection"]
        opposite_pedir = epi_pedir[:1] if "-" in epi_pedir else f"{epi_pedir}-"
        logging.debug("pedir %s, opposite_pedir %s", epi_pedir, opposite_pedir)

        # is there an obvious fmap choice in the fmap dir (eg. spin-echo one)
        fmap1 = layout.get(
            subject=epi.entities["subject"],
            session=epi.entities.get("session", None),
            extension=".nii.gz",
            datatype="fmap",
            PhaseEncodingDirection=epi_pedir,
            ShimSetting=str(epi.entities["ShimSetting"]),
            ImageOrientationPatientDICOM=str(
                epi.entities["ImageOrientationPatientDICOM"]
            ),
        )
        # otherwise look into the modality/datatype folder (B0Field setup)
        if not len(fmap1):
            logging.debug(
                "no fmap found for %s, looking into %s",
                epi.relpath,
                epi.entities["datatype"],
            )

            fmap_query_entities = epi.get_entities().copy()
            fmap_query_entities["echo"] = 1 if epi.entities.get("echo", None) else None
            fmap_query_entities.pop("run", None)

            # get the sbref/epi for that series (1st echo for multi-echo one, itself otherwise)
            fmap1 = layout.get(**fmap_query_entities)
            if use_sbrefs:  # favor sbref if they exists
                fmap1 = (
                    # first look in explicitly named sbref (dwi/func folders)
                    layout.get(**{**fmap_query_entities, "suffix": "sbref"})
                    # then using metadata, if improper suffix use
                    or layout.get(
                        **{
                            **fmap_query_entities,
                            "MultibandAccelerationFactor": Query.NONE,
                        }
                    )
                    or fmap1  # or default to what was found before
                )
            if not len(fmap1) and sloppy > 0:
                #get all the same epi_pedir candidates
                fmap_candidates = layout.get(
                    subject=epi.entities["subject"],
                    session=epi.entities.get("session", None),
                    extension=".nii.gz",
                    datatype="fmap",
                    PhaseEncodingDirection=epi_pedir,
                    )
                # check all the candidates with the defined sloppy rounding error tolerance
                for fmp in fmap_candidates:
                    shim_ok = fmp.entities.get("ShimSetting") == epi.entities.get("ShimSetting")
                    iop_ok = np.allclose(
                        fmp.entities.get("ImageOrientationPatientDICOM", []),
                        epi.entities.get("ImageOrientationPatientDICOM", []),
                        atol=sloppy,
                    )
                    # if both shim and image orientation patient dicom are the same use that fmap
                    if shim_ok and iop_ok:
                        fmap1.append(fmp)

            if not len(fmap1):
                logging.error(
                    f"❌ SBRef/1st-echo epi not found for {epi.relpath}, something went wrong, check BIDS conversion."
                )
                logging.debug(fmap_query_entities)
                continue
        fmap1 = fmap1[0]

        fmap1_is_fmap = fmap1.entities["datatype"] == "fmap"
        logging.info(
            "found same pedir fmap %s as %s", fmap1.relpath, fmap1.entities["datatype"]
        )

        # fetch candidate fieldmaps from the same session but opposite PhaseEncodingDirection
        # can be _epi from fmap, "sbref" or "bold/dwi" from dwi/func folder
        # match fmap1 to avoid matching unrelated things
        fmap_query_base = dict(
            subject=epi.entities["subject"],
            session=epi.entities.get("session", None),
            extension=".nii.gz",
            suffix=[fmap1.entities.get("suffix"), "epi"],
            echo=[fmap1.entities.get("echo", None),None],
            PhaseEncodingDirection=opposite_pedir,
        )
        if fmap1_is_fmap and fmap1.entities.get("acquisition") != None: # add acquisition if that exists already
            fmap_query_base["acquisition"] = fmap1.entities.get("acquisition", None) #BUG: if this is None and it gets added to the fmap_query_base it will return no matches
        # strict match
        logging.info(f"STRICT: this is the query we will try to use {fmap_query_base} + ShimSetting + ImageOrientationPatientDICOM matching ")
        fmaps = layout.get(
            **fmap_query_base,
            ShimSetting=str(epi.entities["ShimSetting"]),
            ImageOrientationPatientDICOM=str(
                epi.entities["ImageOrientationPatientDICOM"]
            ),
        )
        if not fmaps:
            epis_with_shim_mismatch.append(epi)
            logging.warning(
                f"We couldn't find an fieldmap with matching ShimSettings and opposite pedir for: {epi.relpath}. "
                "Including other based on ImageOrientationPatient."
            )
            logging.info(f"Looser: this is the query we will try to use {fmap_query_base} + ImageOrientationPatientDICOM matching ")
            # looser match
            fmaps = layout.get(
                **fmap_query_base,
                ImageOrientationPatientDICOM=str(
                    epi.entities["ImageOrientationPatientDICOM"]
                ),
            )
        if not fmaps:
            logging.error(
                f"We couldn't find an epi fieldmap with matching ImageOrientationPatient and opposite pedir for {epi.relpath}. "
                "Please review manually."
            )
            if sloppy > 0:
                logging.info(f"SLOPPY: this is the query we will try to use {fmap_query_base} + ShimSetthing + ImageOrientationPatientDICOM with rounding error tolerance {sloppy} for matching ")
                all_fmaps = layout.get(**fmap_query_base)
                fmaps = [
                    fmap
                    for fmap in all_fmaps
                    if np.allclose(
                        epi.entities["ImageOrientationPatientDICOM"],
                        fmap.entities["ImageOrientationPatientDICOM"],
                        atol=sloppy,
                    )
                ]
                if not len(fmaps):
                    logging.error(f"Sloppy match gives no {epi.relpath}.")
                    continue
            else:
                epis_with_no_fmap.append(epi)
                continue

        fmaps_sorting_diffs = sorted(
            [(fm, get_index_key(fm) - epi_sorting_value) for fm in fmaps],
            key=itemgetter(1),
        )

        if match_strategy == "first":
            match_fmap = fmaps_sorting_diffs[0][0]
        elif match_strategy == "before":
            delta_for_sbref = datetime.timedelta(seconds=0)
            match_fmap = [
                fm
                for fm, ftd in fmaps_sorting_diffs
                if sorting_value_is_date and ftd <= delta_for_sbref
            ]
            if len(match_fmap):
                match_fmap = match_fmap[-1]
            else:
                logging.warning(
                    f"No fmap matched the {match_strategy} strategy for {epi.relpath}, taking the first match after scan."
                )
                match_fmap = fmaps_sorting_diffs[0][0]
        elif match_strategy == "after":
            # to match the sbref with the corresponding epi, there is a diff of ~11sec, depending on multiband/multi-echo params
            delta_for_sbref = datetime.timedelta(seconds=-30)
            match_fmap = [
                fm
                for fm, ftd in fmaps_sorting_diffs
                if sorting_value_is_date and ftd >= delta_for_sbref
            ]
            if len(match_fmap):
                match_fmap = match_fmap[0]
            else:
                logging.warning(
                    f"No fmap matched the {match_strategy} strategy for {epi.relpath}, taking the first match before scan."
                )
                match_fmap = fmaps_sorting_diffs[-1][0]

        def get_json(nii):
            return nii.get_associations(kind="Metadata")[0].path

        estimator_pair = tuple(sorted([fmap1.relpath, match_fmap.relpath]))
        if estimator_pair in estimator_pairs:
            epi_series_id = estimator_pairs[estimator_pair]
        else:
            fmap1_b0fids = fmap1.entities.get("B0FieldIdentifier", [])
            match_fmap_b0fids = match_fmap.entities.get("B0FieldIdentifier", [])
            if epi_series_id not in match_fmap_b0fids:
                fmaps_append_b0tags.setdefault(get_json(match_fmap), set()).add(epi_series_id)
            if epi_series_id not in fmap1_b0fids:
                fmaps_append_b0tags.setdefault(get_json(fmap1), set()).add(epi_series_id)
            estimator_pairs[estimator_pair] = epi_series_id
        insert_values_in_json(get_json(epi), {"B0FieldSource": epi_series_id})

        # if either fmaps are in the fmap folder, add IntendedFor for retro-compat
        for found_fmap in [match_fmap, fmap1]:
            if found_fmap.entities["datatype"] == "fmap":
                sub_rel_path = str(
                    pathlib.Path(epi.relpath).relative_to(
                        f"./sub-{epi.entities['subject']}/"
                    )
                )
                fmaps_append_intendedfor.setdefault(get_json(found_fmap), set()).add(
                    sub_rel_path
                )

    for fmap_path, b0fieldids in fmaps_append_b0tags.items():
        # avoid lists due to SDCFlows/pybids current limitations: see https://github.com/nipreps/sdcflows/issues/266#issuecomment-1303696056
        b0fieldids = (
            sorted(list(b0fieldids)) if len(b0fieldids) > 1 else b0fieldids.pop()
        )
        new_metas = {"B0FieldIdentifier": b0fieldids}
        print(new_metas)
        if fmap_path in fmaps_append_intendedfor.keys():
            new_metas["IntendedFor"] = sorted(list(fmaps_append_intendedfor[fmap_path]))
        insert_values_in_json(fmap_path, new_metas)


def insert_values_in_json(
    path: pathlib.Path,
    dct: Dict[str, Any],
) -> None:
    logging.info(f"modifying {path} add {dct}")
    with open(path, "r", encoding="utf-8") as fd:
        meta = json.load(fd)
    for tag, values in dct.items():
        meta[tag] = values
    file_mask = os.stat(path)[stat.ST_MODE]
    os.chmod(path, file_mask | stat.S_IWUSR)
    with open(path, "w", encoding="utf-8") as fd:
        fd.write(json_dumps_pretty(meta))
    os.chmod(path, file_mask)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Fill "IntendedFor" field of fieldmaps jsons according to scanning parameters.',
    )
    parser.add_argument("bids_path", type=pathlib.Path, help="BIDS folder to modify")
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
        "--reset",
        action="store_true",
        help="reset existing b0 intent",
    )
    parser.add_argument(
        "--match_strategy",
        choices=["before", "after"],
        default="before",
        help='Strategy to resolve multiple matches: "before"/"after" closest matching in time  ',
    )
    parser.add_argument(
        "--sloppy",
        type=float,
        default=0,
        help="tolerance to allow finding fieldmaps with approximate matching position",
    )

    args = vars(parser.parse_args())
    args["subject"] = args.pop("participant_label") or Query.ANY
    if sessions := args.pop("session_label"):
        args["session"] = sessions
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    fill_b0_meta(**args)
