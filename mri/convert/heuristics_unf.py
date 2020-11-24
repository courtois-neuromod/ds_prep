import os, re
import nibabel.nicom.dicomwrappers as nb_dw
from heudiconv.heuristics import reproin
from heudiconv.heuristics.reproin import (
    OrderedDict,
    create_key,
    get_dups_marked,
    parse_series_spec,
    sanitize_str,
    lgr,
    series_spec_fields,
)


def infotoids(seqinfos, outdir):

    seqinfo = next(seqinfos.__iter__())
    ex_dcm = nb_dw.wrapper_from_file(seqinfo.example_dcm_file_path)

    # pi = str(ex_dcm.dcm_data.ReferringPhysicianName)
    pi = str(seqinfo.referring_physician_name)
    # study_name = str(ex_dcm.dcm_data.StudyDescription)
    study_name = str(seqinfo.study_description)

    patient_name = str(ex_dcm.dcm_data.PatientName)

    study_path = study_name.split("^")

    rema = re.match("(([^_]*)_)?(([^_]*)_)?p([0-9]*)_([a-z]*)([0-9]*)", patient_name)

    locator = os.path.join(pi, *study_path)

    study_name = rema.group(1)
    sub_study_name = rema.group(3)
    subject_id = rema.group(5)
    session_type = rema.group(6)
    session_id = rema.group(7)

    return {
        "locator": locator,
        # Sessions to be deduced yet from the names etc TODO
        "session": session_id,
        "subject": subject_id,
    }


def get_task(s):
    mtch = re.match(".*_task\-([^_]+).*", s.series_id)
    if mtch is not None:
        task = mtch.group(1).split("-")
        if len(task) > 1:
            return task[1]
        return task[0]
    else:
        return None


def get_run(s):
    mtch = re.match(".*run\-([^_]+).*", s.series_id)
    if mtch is not None:
        return mtch.group(1)
    else:
        return None


rec_exclude = [
    "ORIGINAL",
    "PRIMARY",
    "M",
    "MB",
    "ND",
    "MOSAIC",
    "NONE",
    "DIFFUSION",
    "UNI",
]


def get_seq_bids_info(s, ex_dcm):

    seq = {
        "type": "anat",  # by default to make code concise
        "label": None,
    }
    for it in s.image_type[2:]:
        if it not in rec_exclude:
            seq["rec"] = it.lower()

    try:
        pedir = ex_dcm.dcm_data.InPlanePhaseEncodingDirection
        if "COL" in pedir:
            pedir = "AP"
        else:
            pedir = "LR"
        pedir_pos = bool(
            ex_dcm.csa_header["tags"]["PhaseEncodingDirectionPositive"]["items"][0]
        )

        seq["dir"] = pedir if pedir_pos else pedir[::-1]
    except:
        pass

    # label bodypart which are not brain, mainly for spine if we set the dicom fields at the console properly
    bodypart = ex_dcm.dcm_data.get("BodyPartExamined", None)
    if bodypart is not None and bodypart != "BRAIN":
        seq["bp"] = bodypart.lower()

    scan_options = ex_dcm.dcm_data.get("ScanOptions", None)
    image_comments = ex_dcm.dcm_data.get("ImageComments", [])

    # CMRR bold and dwi
    is_sbref = "Single-band reference" in image_comments

    # Anats
    if "localizer" in s.protocol_name.lower():
        seq["label"] = "localizer"
    #        slice_orient = ex_dcm.dcm_data.get([0x0051,0x100e])
    #        if slice_orient is not None:
    #            seq['acq'] = slice_orient.value.lower()
    elif "AAHead_Scout" in s.protocol_name:
        seq["label"] = "scout"
    elif (
        (s.dim4 == 1)
        and ("T1" in s.protocol_name)
        and ("tfl3d1_16ns" in s.sequence_name)
    ):
        seq["label"] = "T1w"
    elif (
        (s.dim4 == 1) and ("T2" in s.protocol_name) and ("spc_314ns" in s.sequence_name)
    ):
        seq["label"] = "T2w"
    elif (
        ("*tfl3d1_16" in s.sequence_name)
        and (s.dim4 == 1)
        and ("mp2rage" in s.protocol_name)
        and not ("memp2rage" in s.protocol_name)
    ):
        seq["label"] = "MP2RAGE"
        if "INV1" in s.series_description:
            seq["inv"] = 1
        elif "INV2" in s.series_description:
            seq["inv"] = 2
        elif "UNI" in s.image_type:
            # seq['acq'] = 'UNI'
            seq["label"] = "UNIT1"  # TODO: validate

    #    elif (s.dim4 == 1) and ('MTw' in s.protocol_name):
    #        seq['label'] = 'MTw'
    #        seq['acq'] = 'off'
    #        if 'On' in s.protocol_name:
    #            seq['acq'] = 'on'

    # GRE acquisition
    elif "*fl3d1" in s.sequence_name:
        seq["label"] = "MTS"
        if "T1w" in s.protocol_name:
            seq["acq"] = "T1w"
        else:
            seq["acq"] = "MTon" if scan_options == "MT" else "MToff"

    elif "tfl2d1" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "B1plusmap"
        seq["acq"] = "flipangle" if "flip angle map" in image_comments else "anat"

    # SWI
    elif (s.dim4 == 1) and ("swi3d1r" in s.sequence_name):
        seq["type"] = "swi"
        if not ("MNIP" in s.image_type):
            seq["label"] = "swi"
        else:
            seq["label"] = "minIP"

    # Siemens or CMRR diffusion sequence, exclude DERIVED (processing at the console)
    elif (
        ("ep_b" in s.sequence_name)
        or ("ez_b" in s.sequence_name)
        or ("epse2d1_110" in s.sequence_name)
    ) and not any(it in s.image_type for it in ["DERIVED", "PHYSIO"]):
        seq["type"] = "dwi"
        seq["label"] = "sbref" if is_sbref else "dwi"

    # CMRR or Siemens functional sequences
    elif "epfid2d1" in s.sequence_name:
        seq["task"] = get_task(s)
        # if no task, this is a fieldmap
        if seq["task"]:
            seq["type"] = "func"
            seq["label"] = "sbref" if is_sbref else "bold"
        else:
            seq["type"] = "fmap"
            seq["label"] = "epi"
            seq["acq"] = "sbref" if is_sbref else "bold"

        seq["run"] = get_run(s)
        if s.is_motion_corrected:
            seq["rec"] = "moco"

    ################## SPINAL CORD PROTOCOL #####################
    elif "spcR_100" in s.sequence_name:
        seq["label"] = "T2w"
    #        seq['bp'] = 'spine'
    elif "*me2d1r3" in s.sequence_name:
        seq["label"] = "T2starmap"

    return seq


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    session: scan index for longitudinal acq
    """

    lgr.info("Processing %d seqinfo entries", len(seqinfo))

    # for s in seqinfo:
    #    print(s)

    info = OrderedDict()
    skipped, skipped_unknown = [], []
    current_run = 0
    run_label = None  # run-
    dcm_image_iod_spec = None
    skip_derived = True

    outtype = ("nii.gz",)
    sbref_as_fieldmap = True  # duplicate sbref in fmap dir to be used by topup
    prefix = ""

    fieldmap_runs = {}

    for s in seqinfo:

        ex_dcm = nb_dw.wrapper_from_file(s.example_dcm_file_path)

        bids_info = get_seq_bids_info(s, ex_dcm)
        print(s)
        print(bids_info)

        # XXX: skip derived sequences, we don't store them to avoid polluting
        # the directory, unless it is the motion corrected ones
        # (will get _rec-moco suffix)
        if (
            skip_derived
            and (s.is_derived or ("MPR" in s.image_type))
            and not s.is_motion_corrected
            and not "UNI" in s.image_type
        ):
            skipped.append(s.series_id)
            lgr.debug("Ignoring derived data %s", s.series_id)
            continue

        seq_type = bids_info["type"]
        seq_label = bids_info["label"]

        if (seq_type == "fmap" and seq_label == "epi") or (
            sbref_as_fieldmap and seq_label == "sbref"
        ):
            pe_dir = bids_info.get("dir", None)
            if not pe_dir in fieldmap_runs:
                fieldmap_runs[pe_dir] = 0
            fieldmap_runs[pe_dir] += 1
            # override the run number
            run_id = fieldmap_runs[pe_dir]

            # duplicate sbref to be used as fieldmap
            if sbref_as_fieldmap and seq_label == "sbref":
                suffix_parts = [
                    "acq-sbref",
                    None if not bids_info.get("ce") else "ce-%s" % bids_info["ce"],
                    None if not pe_dir else "dir-%s" % bids_info["dir"],
                    "run-%02d" % run_id,
                    "epi",
                ]
                suffix = "_".join(filter(bool, suffix_parts))
                template = create_key("fmap", suffix, prefix=prefix, outtype=outtype)
                if template not in info:
                    info[template] = []
                info[template].append(s.series_id)

        show_dir = seq_type in ["fmap", "dwi"]

        # print(bids_info)
        suffix_parts = [
            None if not bids_info.get("task") else "task-%s" % bids_info["task"],
            None if not bids_info.get("acq") else "acq-%s" % bids_info["acq"],
            None if not bids_info.get("ce") else "ce-%s" % bids_info["ce"],
            None
            if not (bids_info.get("dir") and show_dir)
            else "dir-%s" % bids_info["dir"],
            None if not bids_info.get("inv") else "inv-%d" % bids_info["inv"],
            None if not bids_info.get("part") else "part-%s" % bids_info["part"],
            None if not bids_info.get("tsl") else "tsl-%d" % bids_info["tsl"],
            None if not bids_info.get("loc") else "loc-%s" % bids_info["loc"],
            None if not bids_info.get("run") else "run-%02d" % int(bids_info["run"]),
            None if not bids_info.get("bp") else "bp-%s" % bids_info["bp"],
            None if not bids_info.get("echo") else "echo-%d" % int(bids_info["echo"]),
            seq_label,
        ]
        # filter those which are None, and join with _
        suffix = "_".join(filter(bool, suffix_parts))

        # if "_Scout" in s.series_description or \
        #        (seqtype == 'anat' and seqtype_label and seqtype_label.startswith('scout')):
        #    outtype = ('dicom',)
        # else:
        #    outtype = ('nii.gz', 'dicom')

        template = create_key(seq_type, suffix, prefix=prefix, outtype=outtype)

        # we wanted ordered dict for consistent demarcation of dups
        if template not in info:
            info[template] = []
        else:
            # maybe images are exported with different reconstruction parameters.
            if bids_info.get("rec") and not any([]):
                # insert the rec-
                suffix_parts.insert(7, "rec-%s" % bids_info["rec"])
                # filter those which are None, and join with _
                suffix = "_".join(filter(bool, suffix_parts))
                template = create_key(seq_type, suffix, prefix=prefix, outtype=outtype)
                info[template] = []

        info[template].append(s.series_id)

    if skipped:
        lgr.info("Skipped %d sequences: %s" % (len(skipped), skipped))
    if skipped_unknown:
        lgr.warning(
            "Could not figure out where to stick %d sequences: %s"
            % (len(skipped_unknown), skipped_unknown)
        )

    info = get_dups_marked(info)  # mark duplicate ones with __dup-0x suffix

    info = dict(
        info
    )  # convert to dict since outside functionality depends on it being a basic dict

    for k, i in info.items():
        print(k, i)
    return info
