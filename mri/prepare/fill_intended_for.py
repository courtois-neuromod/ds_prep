import sys, os
import shutil, stat
import pathlib
import re, fnmatch
from bids import BIDSLayout
from bids.layout import Query
import json
import logging
import argparse
import numpy as np
import datetime
from operator import itemgetter
from heudiconv.utils import json_dumps_pretty

PYBIDS_CACHE_PATH = ".pybids_cache"

def _load_bidsignore_(bids_root):
    """Load .bidsignore file from a BIDS dataset, returns list of regexps"""
    bids_root = pathlib.Path(bids_root)
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

def fill_b0_meta(bids_path, participant_label=None, session_label=None, force_reindex=False, match_strategy='before', sloppy=False, **kwargs):
    path = os.path.abspath(bids_path)
    pybids_cache_path = os.path.join(path, PYBIDS_CACHE_PATH)

    layout = BIDSLayout(
        path,
        database_path=pybids_cache_path,
        reset_database=force_reindex,
        validate=False,
        ignore=_load_bidsignore_(path),
        index_metadata=True,
    )
    extra_filters = {}
    if participant_label:
        extra_filters["subject"] = participant_label
    if session_label:
        extra_filters["session"] = session_label

    base_entities = dict(
        suffix=["bold", "dwi"],
        extension=".nii.gz",
        **extra_filters
        )
    epis = layout.get(part='mag', **base_entities) + \
            layout.get(part=Query.NONE, **base_entities)
    
    fmaps_to_modify = {}

    epis_with_no_fmap = []
    epis_with_shim_mismatch = []
    
    for epi in epis:
        logging.info(f"matching fmap for {epi.path}")
        epi_series_id = os.path.basename(epi.path).split('.')[0].split('_echo')[0].split('_part')[0].replace('-', '_')
        epi_b0fieldsource = epi.entities.get('B0FieldSource', None)
        epi_scan_time = datetime.datetime.strptime(
            epi.tags["AcquisitionTime"].value, "%H:%M:%S.%f"
        )
        
        #if epi_b0fieldsource and epi_series_id in epi_b0fieldsource:
            # that series was already assigned a fieldmap
            #continue
        epi_pedir = epi.entities["PhaseEncodingDirection"]
        opposite_pedir = epi_pedir[-1:] if '-' in epi_pedir else f"{epi_pedir}-"

        sbref = layout.get(**{
            **epi.get_entities(),
            'suffix':'sbref',
            'echo': 1 if epi.entities.get('echo', None) else None # 
        })
        assert len(sbref)==1, "There should be a single SBRef for each epi"
        if not sbref:
            logging.error(f"SBref not found for {epi.path}, something went wrong, check BIDS conversion.")
            continue
        sbref = sbref[0]
        
        #if epi_series_id in sbref.entities.get('B0FieldIdentifier',[]):
            # that series was already assigned a fieldmap
        #    continue
            
        # fetch candidate fieldmaps from the same session            
        fmap_query_base = dict(
            suffix=["epi", "sbref"],
            extension=".nii.gz",
            acquisition=["sbref", "sbrefEcho1"], # get first echo
            subject=epi.entities["subject"],
            session=epi.entities.get("session", None),
#            echo=1 if epi.entities.get("echo", None) else None, # get first echo
            PhaseEncodingDirection=opposite_pedir,
        )
        
        # strict match
        fmaps = layout.get(
            **fmap_query_base,
            ShimSetting=str(epi.entities['ShimSetting']),
            ImageOrientationPatientDICOM=str(epi.entities['ImageOrientationPatientDICOM']),
        )
        logging.debug("query fmaps" + str(fmap_query_base))
        if not fmaps:
            epis_with_shim_mismatch.append(epi)
            logging.warning(
                f"We couldn't find an fieldmap with matching ShimSettings and opposite pedir for: {epi.path}. "
                "Including other based on ImageOrientationPatient."
            )
            # looser match
            fmaps = layout.get(
                **fmap_query_base,
                ImageOrientationPatientDICOM=str(epi.entities['ImageOrientationPatientDICOM']),
            )
        if not fmaps:
            logging.error(
                f"We couldn't find an epi fieldmaps with matching ImageOrientationPatient and opposite pedir for {epi.path}. "
                "Please review manually.")
            if sloppy > 0:
                all_fmaps = layout.get(
                    **fmap_query_base,
                )
                fmaps = [fmap for fmap in all_fmaps
                         if np.allclose(epi.entities['ImageOrientationPatientDICOM'],
                                        fmap.entities['ImageOrientationPatientDICOM'],
                                        atol=sloppy)]
                if not len(fmaps):
                    logging.error(f"Sloppy match gives no {epi.path}.")
                    continue
            else:
                epis_with_no_fmap.append(epi)
                continue

            
        fmaps_time_diffs = sorted(
            [(
                fm,
                datetime.datetime.strptime(
                    fm.entities["AcquisitionTime"], "%H:%M:%S.%f"
                ) - epi_scan_time
            )
             for fm in fmaps ],
            key=itemgetter(1),
        )
        
        if match_strategy == "before":
            delta_for_sbref = datetime.timedelta(seconds=0)
            match_fmap = [
                fm for fm, ftd in fmaps_time_diffs if ftd <= delta_for_sbref
                ]
            if len(match_fmap):
                match_fmap = match_fmap[-1]
            else:
                logging.warning(
                    f"No fmap matched the {match_strategy} strategy for {epi.path}, taking the first match after scan."
                )
                match_fmap = fmaps_time_diffs[0][0]
        elif match_strategy == "after":
            # to match the sbref with the corresponding epi, there is a diff of ~11sec, depending on multiband/multi-echo params
            delta_for_sbref = datetime.timedelta(seconds=-30)
            match_fmap = [
                    fm for fm, ftd in fmaps_time_diffs if ftd >= delta_for_sbref
            ]
            if len(match_fmap):
                match_fmap = match_fmap[0]
            else:
                logging.warning(
                    f"No fmap matched the {match_strategy} strategy for {epi.path}, taking the first match before scan."
                )
                match_fmap = fmaps_time_diffs[-1][0]


        if ("B0FieldIdentifier" not in match_fmap.entities) or (
                epi_series_id not in match_fmap.tags.get("B0FieldIdentifier").value
        ):
            fmap_json_path = match_fmap.get_associations(kind='Metadata')[0].path
            logging.info(f"_______________{fmap_json_path}")

            if fmap_json_path not in fmaps_to_modify:
                fmaps_to_modify[fmap_json_path] = []
            if epi_series_id not in fmaps_to_modify[fmap_json_path]:
                fmaps_to_modify[fmap_json_path].append(epi_series_id)

        insert_values_in_json(
            sbref.get_associations(kind='Metadata')[0].path,
            {'B0FieldIdentifier': epi_series_id,
             'B0FieldSource': epi_series_id}
        )
        insert_values_in_json(
            epi.get_associations(kind='Metadata')[0].path,
            {'B0FieldSource': epi_series_id}
        )
    for fmap_path, b0fieldids in fmaps_to_modify.items():
        #avoid lists due to SDCFlows/pybids current limitations: see https://github.com/nipreps/sdcflows/issues/266#issuecomment-1303696056
        b0fieldids = b0fieldids if len(b0fieldids)>1 else b0fieldids[0]
        insert_values_in_json(
            fmap_path,
            {'B0FieldIdentifier': b0fieldids}
        )
            
def insert_values_in_json(path, dct):
    logging.info(f"modifying {path} add {dct}")
    with open(path, "r", encoding="utf-8") as fd:
        meta = json.load(fd)
    for tag, values in dct.items():
        meta[tag] = values #sorted(list(set(meta.get(tag,[])+values)))
    file_mask = os.stat(path)[stat.ST_MODE]
    try:
        os.chmod(path, file_mask | stat.S_IWUSR)
    except PermissionError:
        pass
    with open(path, "w", encoding="utf-8") as fd:
        fd.write(json_dumps_pretty(meta))
    try:
        os.chmod(path, file_mask)
    except PermissionError:
        pass

def get_candidate_fmaps(layout, epi, match_shim=True, sloppy=0, non_fmap=True):
    
    epi_pedir = epi.entities["PhaseEncodingDirection"]
    opposite_pedir = epi_pedir[0] if '-' in epi_pedir else f"{epi_pedir}-"
        
    fmap_query_base = dict(
        extension=".nii.gz",
        subject=epi.entities["subject"],
        session=epi.entities.get("session", None),
        echo=1 if epi.entities.get("echo", None) else None, # get first echo
        #PhaseEncodingDirection=opposite_pedir,
    )

    dtype_suffix_acq_part_comb = [
        ('fmap', 'epi', 'sbref', None),
        ('fmap', 'epi', Query.NONE, None)]
    if non_fmap:
        dtype_suffix_acq_part_comb.extend([
        (epi.entities['datatype'], 'sbref', None, Query.NONE),
        (epi.entities['datatype'], 'sbref', None, 'mag')])
    
    fmaps = sum([
        layout.get(
            **fmap_query_base,
            datatype=dtype,
            suffix=suffix,
            acquisition=acq,
            part=part,
            ShimSetting=str(epi.entities['ShimSetting']) if match_shim else Query.ANY,
            ImageOrientationPatientDICOM=str(epi.entities['ImageOrientationPatientDICOM']) if not sloppy else Query.ANY,
        ) for dtype, suffix, acq, part in dtype_suffix_acq_part_comb ],[])
    
    if sloppy > 0:
        # find fmaps with close enough patient position
        fmaps = [fmap for fmap in fmaps
                 if np.allclose(epi.entities['ImageOrientationPatientDICOM'],
                                fmap.entities['ImageOrientationPatientDICOM'],
                                atol=sloppy)]
        if not len(fmaps):
            logging.error(f"Sloppy match gives no {epi.path}.")

    # only get 2 images with opposed pedir
    fmaps_match_pe_pos = [
        fm for fm in fmaps
        if "-" not in fm.tags["PhaseEncodingDirection"].value
    ]
    fmaps_match_pe_neg = [
        fm for fm in fmaps if "-" in fm.tags["PhaseEncodingDirection"].value
    ]

    return fmaps_match_pe_pos, fmaps_match_pe_neg

def fill_intended_for(bids_path, participant_label=None, session_label=None, force_reindex=False, match_strategy='before', sloppy=False, **kwargs):
    path = os.path.abspath(bids_path)
    pybids_cache_path = os.path.join(path, PYBIDS_CACHE_PATH)

    layout = BIDSLayout(
        path,
        database_path=pybids_cache_path,
        reset_database=force_reindex,
        validate=False,
        ignore=_load_bidsignore_(path),
    )
    extra_filters = {}
    if participant_label:
        extra_filters["subject"] = participant_label
    if session_label:
        extra_filters["session"] = session_label


    base_entities = dict(
        suffix=["bold", "dwi"],
        extension=".nii.gz",
        **extra_filters
        )

    epis = layout.get(**base_entities, part='mag') + \
            layout.get(**base_entities, part=Query.NONE)
    logging.info(f"found {len(epis)} runs")
    json_to_modify = dict()

    epis_with_no_fmap = []
    epis_with_shim_mismatch = []

    for epi in epis:
        epi_series_id = os.path.basename(epi.path).split('.')[0].split('_echo')[0].split('_part')[0].replace('-', '_')
        epi_b0fieldsource = epi.entities.get('B0FieldSource', None)
        epi_scan_time = datetime.datetime.strptime(
            epi.tags["AcquisitionTime"].value, "%H:%M:%S.%f"
        )

        fmaps_match_pe_pos, fmaps_match_pe_neg = get_candidate_fmaps(layout, epi, sloppy=sloppy, non_fmap=False)
        if not fmaps_match_pe_pos or not fmaps_match_pe_neg:
            fmaps_match_pe_pos, fmaps_match_pe_neg = get_candidate_fmaps(layout, epi, sloppy=sloppy, non_fmap=False, match_shim=False)
        if not fmaps_match_pe_pos or not fmaps_match_pe_neg:
            logging.error(f"no matching fieldmaps for: {epi.relpath}")
            continue

        for fmap in [fmaps_match_pe_pos, fmaps_match_pe_neg]:
            fmaps_time_diffs = sorted(
                [
                    (
                        fm,
                        datetime.datetime.strptime(
                            fm.tags.get("AcquisitionTime").value, "%H:%M:%S.%f"
                        )
                        - epi_scan_time,
                    )
                    for fm in fmap
                ],
                key=itemgetter(1),
            )

            if match_strategy == "before":
                delta_for_sbref = datetime.timedelta(seconds=0)
                match_fmap = [
                    fm for fm, ftd in fmaps_time_diffs if ftd <= delta_for_sbref
                ]
                if len(match_fmap):
                    match_fmap = match_fmap[-1]
                else:
                    logging.warning(
                        f"No fmap matched the {match_strategy} strategy for {epi.path}, taking the first match after scan."
                    )
                    match_fmap = fmaps_time_diffs[0][0]
            elif match_strategy == "after":
                # to match the sbref with the corresponding epi, there is a diff of ~11sec
                delta_for_sbref = datetime.timedelta(seconds=-15)
                match_fmap = [
                    fm for fm, ftd in fmaps_time_diffs if ftd >= delta_for_sbref
                ]
                if len(match_fmap):
                    match_fmap = match_fmap[0]
                else:
                    logging.warning(
                        f"No fmap matched the {match_strategy} strategy for {epi.path}, taking the first match before scan."
                    )
                    match_fmap = fmaps_time_diffs[-1][0]
            if ("IntendedFor" not in match_fmap.tags) or (
                epi.path not in match_fmap.tags.get("IntendedFor").value
            ):
                fmap_json_path = match_fmap.get_associations()[0].path
                if fmap_json_path not in json_to_modify:
                    json_to_modify[fmap_json_path] = []
                json_to_modify[fmap_json_path].append(
                    os.path.relpath(epi.path, epi.path.split("ses-")[0])
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

    if len(epis_with_no_fmap):
        epis_with_no_fmap_path = [
            os.path.relpath(epi.path, path) for epi in epis_with_no_fmap
        ]
        logging.error(
            "No phase-reversed fieldmap was found for the following files:\n"
            + "\n".join(epis_with_no_fmap_path)
        )
    if len(epis_with_shim_mismatch):
        epis_with_shim_mismatch_paths = [
            os.path.relpath(epi.path, path) for epi in epis_with_shim_mismatch
        ]
        logging.error(
            "No phase-reversed with matching ShimSettings was found following files:\n"
            + "\n".join(epis_with_shim_mismatch_paths)
        )


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
    parser.add_argument(
        "--b0-field-id",
        action="store_true",
        help="fill new BIDS B0FieldIdentifier instead of IntendedFor",
    )


    parser.add_argument(
        "--sloppy",
        type=float,
        default=0,
        help="tolerance to allow finding fieldmaps with approximate matching position",
    )

    return parser.parse_args()


if __name__ == "__main__":
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=LOGLEVEL)
    
    args = parse_args()
    if args.b0_field_id:
        fill_b0_meta(**vars(args))
    else:
        fill_intended_for(**vars(args))
    
