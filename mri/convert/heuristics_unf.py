import os, re, glob
from frozendict import frozendict
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

def load_example_dcm(seqinfo):
    ex_dcm_path = sorted(glob.glob(os.path.join('/tmp', 'heudiconv*', '*', seqinfo.dcm_dir_name, seqinfo.example_dcm_file)))[0]
    return nb_dw.wrapper_from_file(ex_dcm_path)

def custom_seqinfo(wrapper, series_files):
    #print('calling custom_seqinfo', wrapper, series_files)

    pedir_pos = None
    if hasattr(wrapper, 'csa_header'):
        pedir_pos = wrapper.csa_header["tags"]["PhaseEncodingDirectionPositive"]["items"]
        pedir_pos = pedir_pos[0] if len(pedir_pos) else None
    custom_info = frozendict({
        'patient_name': wrapper.dcm_data.PatientName,
        'pe_dir': wrapper.dcm_data.get('InPlanePhaseEncodingDirection', None),
        'pe_dir_pos': pedir_pos,
        'body_part': wrapper.dcm_data.get("BodyPartExamined", None),
        'scan_options': str(wrapper.dcm_data.get("ScanOptions", None)),
        'image_comments': wrapper.dcm_data.get("ImageComments", ""),
        'slice_orient': str(wrapper.dcm_data.get([0x0051,0x100e]).value),
        'echo_number': str(wrapper.dcm_data.get("EchoNumber", None)),
        'rescale_slope': wrapper.dcm_data.get("RescaleSlope", None),
    })
    return custom_info

def infotoids(seqinfos, outdir):

    seqinfo = next(seqinfos.__iter__())

    #ex_dcm = load_example_dcm(seqinfo)

    pi = str(seqinfo.referring_physician_name)
    study_name = str(seqinfo.study_description)
    patient_name = str(seqinfo.custom['patient_name'])

    study_path = study_name.split("^")

    rema = re.match("(([^_]*)_)?(([^_]*)_)?p([0-9]*)_([a-zA-Z]*)([0-9]*)", patient_name)
    if rema is None:
        rema = re.match("(([^_]*)_)?(([^_]*)_)?(dev)_([a-zA-Z]*)([0-9]*)", patient_name)
    if rema:
        study_name = rema.group(1)
        sub_study_name = rema.group(3)
        subject_id = rema.group(5)
        session_type = rema.group(6)
        session_id = rema.group(7)

    if rema is None:
        rema = re.match("(([^_]*)_)?([a-zA-Z0-9]*)_([a-zA-Z0-9]*)", patient_name)
        study_name = rema.group(2)
        subject_id = rema.group(3)
        session_id = rema.group(4)

    locator = os.path.join(pi, *study_path)

    return {
#        "locator": locator,
        # Sessions to be deduced yet from the names etc TODO
        "session": session_id,
        "subject": subject_id,
    }


def get_task(s):
    mtch = re.match(".*_task\-([^_]+).*", s.series_id)
    if mtch is None:
        mtch = re.match(".*\-task_([^_]+).*", s.series_id)# for floc messup
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
    "P",
    "MB",
    "ND",
    "MOSAIC",
    "NONE",
    "DIFFUSION",
    "UNI",
] + [f"TE{i}" for i in range(9)]


def get_seq_bids_info(s):

    seq = {
        "type": "anat",  # by default to make code concise
        "label": None,
    }

    seq_extra = {}
    for it in s.image_type[2:]:
        if it not in rec_exclude:
            seq_extra["rec"] = it.lower()
    seq_extra["part"] = "mag" if "M" in s.image_type else ("phase" if "P" in s.image_type else None)
    
    try:
        pedir = s.custom['pe_dir']
        if "COL" in pedir:
            pedir = "AP"
        else:
            pedir = "LR"
        pedir_pos = bool(
            s.custom['pe_dir_pos']
        )

        seq["dir"] = pedir if pedir_pos else pedir[::-1]
    except:
        pass

    # label bodypart which are not brain, mainly for spine if we set the dicom fields at the console properly
    bodypart = s.custom['body_part'] #ex_dcm.dcm_data.get("BodyPartExamined", None)
    if bodypart is not None and bodypart != "BRAIN":
        seq["bp"] = bodypart.lower()
        print(seq)

    scan_options = s.custom['scan_options'] #ex_dcm.dcm_data.get("ScanOptions", None)
    image_comments = s.custom['image_comments'] #ex_dcm.dcm_data.get("ImageComments", [])

    # CMRR bold and dwi
    is_sbref = "Single-band reference" in image_comments
    print(s, is_sbref)

    # Anats
    if "localizer" in s.protocol_name.lower():
        seq["label"] = "localizer"
        slice_orient = s.custom['slice_orient'] #ex_dcm.dcm_data.get([0x0051,0x100e]) 
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
        seq["mt"] = "on" if scan_options == "MT" else "off"
        # do not work for multiple flip-angle, need data to find how to detect index
        seq["flip"] = 2 if 'T1w' in s.series_id else 1

    elif "tfl2d1" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "TB1TFL"
        seq["acq"] = "famp" if "flip angle map" in image_comments else "anat"

    elif "fm2d2r" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "phasediff" if "phase" in s.image_type else "magnitude%d"%s.custom['echo_number']     
        
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

        # dumb far-fetched heuristics, no info in dicoms see https://github.com/CMRR-C2P/MB/issues/305
        seq_extra["part"] = 'phase' if s.custom['rescale_slope'] else 'mag'


    # CMRR or Siemens functional sequences
    elif "epfid2d" in s.sequence_name:
        seq["task"] = get_task(s)

        # if no task, this is a fieldmap
        if "AP" in s.series_id and not seq["task"]:
            seq["type"] = "fmap"
            seq["label"] = "epi"
            seq["acq"] = "sbref" if is_sbref else "bold"
        else:
            seq["type"] = "func"
            seq["label"] = "sbref" if is_sbref else "bold"

        seq["run"] = get_run(s)
        if s.is_motion_corrected:
            seq["rec"] = "moco"

            
    ################## SPINAL CORD PROTOCOL #####################
    elif "spcR_100" in s.sequence_name:
        seq["label"] = "T2w"
    #        seq['bp'] = 'spine'
    elif "*me2d1r3" in s.sequence_name:
        seq["label"] = "T2starw"

    if seq["label"] == "sbref" and "part" in seq:
        del seq["part"]
        
    return seq, seq_extra


def generate_bids_key(seq_type, seq_label, prefix, bids_info, show_dir=False, outtype=("nii.gz",), **bids_extra):
    bids_info.update(bids_extra)
    suffix_parts = [
        None if not bids_info.get("task") else "task-%s" % bids_info["task"],
        None if not bids_info.get("acq") else "acq-%s" % bids_info["acq"],
        None if not bids_info.get("ce") else "ce-%s" % bids_info["ce"],
        None
        if not (bids_info.get("dir") and show_dir)
        else "dir-%s" % bids_info["dir"],
        None if not bids_info.get("rec") else "rec-%s" % bids_info["rec"],
        None if not bids_info.get("inv") else "inv-%d" % bids_info["inv"],
        None if not bids_info.get("tsl") else "tsl-%d" % bids_info["tsl"],
        None if not bids_info.get("loc") else "loc-%s" % bids_info["loc"],
        None if not bids_info.get("bp") else "bp-%s" % bids_info["bp"],
        None if not bids_info.get("run") else "run-%02d" % int(bids_info["run"]),
        None if not bids_info.get("echo") else "echo-%d" % int(bids_info["echo"]),
        None if not bids_info.get("flip") else "flip-%d" % int(bids_info["flip"]),
        None if not bids_info.get("mt") else "mt-%s" % bids_info["mt"],
        None if not bids_info.get("part") else "part-%s" % bids_info["part"],
        seq_label,
    ]
    # filter those which are None, and join with _
    suffix = "_".join(filter(bool, suffix_parts))
    
    return create_key(seq_type, suffix, prefix=prefix, outtype=outtype)


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    session: scan index for longitudinal acq
    """

    #lgr.info("Processing %d seqinfo entries", len(seqinfo))
    #lgr.info(seqinfo)

    info = OrderedDict()
    skipped, skipped_unknown = [], []
    current_run = 0
    run_label = None  # run-
    dcm_image_iod_spec = None
    skip_derived = True

    outtype = ("nii.gz",)
    sbref_as_fieldmap = True  # duplicate sbref in fmap dir to be used by topup
    #sbref_as_fieldmap = False # sbref as fieldmaps is still required to use fMRIPrep LTS.
    prefix = ""

    fieldmap_runs = {}
    all_bids_infos = {}

    for s in seqinfo:
        
        #ex_dcm = load_example_dcm(s)

        bids_info, bids_extra = get_seq_bids_info(s)
        all_bids_infos[s.series_id] = (bids_info, bids_extra)

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

        if (seq_type == "fmap" and seq_label == "epi" and bids_extra['part']=='phase' and seq_label=='bold'):
            continue
        
        if ((seq_type == "fmap" and seq_label == "epi") or
            (sbref_as_fieldmap and seq_label == "sbref" and seq_type=='bold')
        ) and bids_info.get("part") in ["mag", None]:
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

        show_dir = seq_type in ["fmap", "dwi"] and not seq_label=='TB1TFL'

        template = generate_bids_key(seq_type, seq_label, prefix, bids_info, show_dir, outtype)

        if template not in info:
            info[template] = []
        info[template].append(s.series_id)
        

    if skipped:
        lgr.info("Skipped %d sequences: %s" % (len(skipped), skipped))
    if skipped_unknown:
        lgr.warning(
            "Could not figure out where to stick %d sequences: %s"
            % (len(skipped_unknown), skipped_unknown)
        )

    info = dedup_bids_extra(info, all_bids_infos)
    info = get_dups_marked(info)  # mark duplicate ones with __dup-0x suffix

    info = dict(
        info
    )  # convert to dict since outside functionality depends on it being a basic dict

    for k, i in info.items():
        lgr.info(f"{k} {i}")

    return info


def dedup_bids_extra(info, bids_infos):
    # add `rec-` or `part-` to dedup series originating from the same acquisition
    info = info.copy()
    for template, series_ids in list(info.items()):
        if len(series_ids) >= 2:
            lgr.warning("Detected %d run(s) for template %s: %s",
                        len(series_ids), template[0], series_ids)

            for extra in ["rec", "part"]:
                
                bids_extra_values = [bids_infos[sid][1].get(extra) for sid in series_ids]

                if len(set(bids_extra_values)) < 2:
                    continue #does not differentiate series

                lgr.info(f"dedup series using {extra}")

                for sid in list(series_ids): #need a copy of list because we are removing elements in that loop

                    series_bids_info, series_bids_extra = bids_infos[sid]

                    new_template = generate_bids_key(
                        series_bids_info["type"],
                        series_bids_info["label"],
                        "",
                        series_bids_info,
                        show_dir=series_bids_info["type"] in ["fmap", "dwi"],
                        outtype=("nii.gz",),
                        **{extra: series_bids_extra.get(extra)})

                    if new_template not in info:
                        info[new_template] = []
                    info[new_template].append(sid)
                    info[template].remove(sid)
                    if not len(info[template]):
                        del info[template]
                break
    return info
