import os, re, glob
from frozendict import frozendict
import nibabel.nicom.dicomwrappers as nb_dw
from heudiconv.heuristics import reproin
from heudiconv.heuristics.reproin import (
    create_key,
    get_dups_marked,
    parse_series_spec,
    sanitize_str,
    lgr,
    series_spec_fields,
)
from collections import OrderedDict
import pydicom

def load_example_dcm(seqinfo):
    ex_dcm_path = sorted(glob.glob(os.path.join('/tmp', 'heudiconv*', '*', seqinfo.dcm_dir_name, seqinfo.example_dcm_file)))[0]
    return nb_dw.wrapper_from_file(ex_dcm_path)


XA_SDI_TAGS = {
    'image_type_text': (0x0021, 0x1175),
    'pedir_pos': (0x0021,0x111c),
    'ice_dims': (0x0021,0x1106),
    'image_history': (0x0021,0x1176),
    }

def custom_seqinfo(wrapper, series_files):
    #print('calling custom_seqinfo', wrapper, series_files)

    image_history = ice_dims = pedir_pos = image_type_text = None
    pe_dir = wrapper.dcm_data.get('InPlanePhaseEncodingDirection', None)
    slice_orient = wrapper.dcm_data.get([0x0051,0x100e])
    receive_coil = wrapper.dcm_data.get((0x0051,0x100f))
    receive_coil = receive_coil.value if receive_coil else None
    custom_info = {}
    flip_angle =  wrapper.dcm_data.FlipAngle if 'FlipAngle' in wrapper.dcm_data else None
    scan_options = str(wrapper.dcm_data.get("ScanOptions", None)),
    
    if hasattr(wrapper, 'csa_header'):
        pedir_pos = wrapper.csa_header["tags"]["PhaseEncodingDirectionPositive"]["items"]
        pedir_pos = pedir_pos[0] if len(pedir_pos) else None
        image_history = wrapper.csa_header['tags']['ImageHistory']['items']
        ice_dims = wrapper.csa_header['tags']['ICE_Dims']['items'][0]
    elif hasattr(wrapper, 'frames'):
        # XA multiframe enhanced dicoms
        if (0x0021, 0x11fe) in wrapper.frames[0]:
            frame0_sdi_private_info = wrapper.frames[0][(0x0021, 0x11fe)][0]
            for key, code in XA_SDI_TAGS.items():
                if code in frame0_sdi_private_info:
                    val = frame0_sdi_private_info.get(code)
                    custom_info[key] = None
                    if val:
                        val = val.value
                        if isinstance(val, pydicom.multival.MultiValue):
                            val = tuple(val)
                        else:
                            val = str(val)
                        print(val, val.__class__)
                        custom_info[key] = val
        if 'MRFOVGeometrySequence' in wrapper.shared:
            pe_dir = wrapper.shared.MRFOVGeometrySequence[0].get('InPlanePhaseEncodingDirection')
        if 'MRReceiveCoilSequence' in wrapper.shared:
            receive_coil = wrapper.shared.MRReceiveCoilSequence[0].get('ReceiveCoilName')
        if 'MRTimingAndRelatedParametersSequence' in wrapper.shared:
            flip_angle = wrapper.shared.MRTimingAndRelatedParametersSequence[0].get('FlipAngle')
        if (0x0021, 0x10fe) in wrapper.shared:
            scan_options = wrapper.shared[(0x0021, 0x10fe)][0].get((0x0021, 0x105c))
            scan_options = tuple(scan_options) if scan_options
            
            
    custom_info = {
        'patient_name': wrapper.dcm_data.PatientName,
        'pe_dir': pe_dir,
        'pe_dir_pos': bool(int(pedir_pos)) if pedir_pos else None,
        'pulse_sequence_name':wrapper.dcm_data.get('PulseSequenceName', None),
        'body_part': wrapper.dcm_data.get("BodyPartExamined", None),
        'scan_options': scan_options,
        'image_comments': wrapper.dcm_data.get("ImageComments", ""),
        'slice_orient': str(slice_orient.value) if slice_orient else None,
        'echo_number': str(wrapper.dcm_data.get("EchoNumber", None)),
        'flip_angle': flip_angle,
        'rescale_slope': wrapper.dcm_data.get("RescaleSlope", None),
        'receive_coil': receive_coil,
        'image_history': ';'.join(filter(len, image_history)) if image_history else None,
        'ice_dims': ice_dims,
        'image_type_text': image_type_text,
        } | custom_info

    print(custom_info)
    custom_info = frozendict(custom_info)


    return custom_info

def infotoids(seqinfos, outdir):

    seqinfo = next(seqinfos.__iter__())

    #ex_dcm = load_example_dcm(seqinfo)

    pi = str(seqinfo.referring_physician_name)
    study_name = str(seqinfo.study_description)
    patient_name = str(seqinfo.custom['patient_name'])

    study_path = study_name.split("^")

    study_name = 'unknown'
    subject_id = 'unknown'
    session_id = 'unknown'
 
    rema = re.match("(([^_]*)_)?(([^_]*)_)?p([0-9]*)_([a-zA-Z0-9]*)([0-9]{3})", patient_name)
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
        if rema:
            study_name = rema.group(2)
            subject_id = rema.group(3)
            session_id = rema.group(4)
        else:
            subject_id = patient_name.split(' ')[-1]
            session_id = None

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

    image_type = s.custom['image_type_text'] or s.image_type
    
    seq_extra = {}
    for it in image_type[2:]:
        if it not in rec_exclude:
            seq_extra["rec"] = it.lower()
    seq_extra["part"] = "mag" if "M" in image_type else ("phase" if "P" in image_type else None)

    try:
        pedir = s.custom['pe_dir']
        if "COL" in pedir[:3]:
            pedir = "AP"
        else:
            pedir = "LR"
        pedir_pos = bool(
            int(s.custom['pedir_pos'])
        )

        seq["dir"] = pedir if pedir_pos else pedir[::-1]
    except Exception as e:
        print(s, e)
        pass

    # label bodypart which are not brain, mainly for spine if we set the dicom fields at the console properly
    bodypart = s.custom['body_part'] #ex_dcm.dcm_data.get("BodyPartExamined", None)
    if bodypart is not None and bodypart != "BRAIN":
        seq["bp"] = bodypart.lower()
        print(seq)

    scan_options = s.custom['scan_options']
    image_comments = s.custom['image_comments']

    # CMRR bold and dwi
    is_sbref = "Single-band reference" in image_comments

    if s.custom['ice_dims'] and s.custom['ice_dims'][0] != 'X' and not s.is_derived:
        seq['rec'] = 'uncombined'
    # Anats
    if "localizer" in s.protocol_name.lower():
        seq["label"] = "localizer"
        slice_orient = s.custom['slice_orient'] #ex_dcm.dcm_data.get([0x0051,0x100e])
        if slice_orient is not None:
            seq_extra['acq'] = slice_orient.lower()
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
        elif "UNI" in image_type:
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
        seq["mt"] = "on" if ("MT" in scan_options) else "off"
        # do not work for multiple flip-angle, need data to find how to detect index
        seq["flip"] = 2 if ('T1w' in s.series_id or 'MTT1' in s.series_id) else 1

    elif "tfl2d1" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "TB1TFL"
        seq["acq"] = "famp" if "flip angle map" in image_comments else "anat"
    elif "epse2d1_9" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "TB1EPI"
        seq["flip"] = 1 if s.custom['flip_angle'] < 90 else 2
        

    elif "fm2d2r" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "phasediff" if "phase" in image_type else "magnitude%d"%s.custom['echo_number']

    # SWI
    elif (s.dim4 == 1) and ("swi3d1r" in s.sequence_name):
        seq["type"] = "swi"
        if not ("MNIP" in image_type):
            seq["label"] = "swi"
        else:
            seq["label"] = "minIP"

    # Siemens or CMRR diffusion sequence, exclude DERIVED (processing at the console)
    elif (
        ("ep_b" in s.sequence_name)
        or ("ez_b" in s.sequence_name)
        or ("epse2d1" in s.sequence_name)
    ) and not any(it in image_type for it in ["DERIVED", "PHYSIO"]):
        seq["type"] = "dwi"
        seq["label"] = "sbref" if is_sbref else "dwi"

        # dumb far-fetched heuristics, no info in dicoms see https://github.com/CMRR-C2P/MB/issues/305
        seq_extra["part"] = 'phase' if s.custom['rescale_slope'] else 'mag'


    # CMRR or Siemens functional sequences
    elif "epfid2d" in s.sequence_name:
        seq["task"] = get_task(s)

        # if no task, this is a fieldmap
#        if "AP" in s.series_id and not seq["task"]:
#            seq["type"] = "fmap"
#            seq["label"] = "epi"
#            seq["acq"] = "sbref" if is_sbref else "bold"
#        else:
        seq["type"] = "func"
        seq["label"] = "sbref" if is_sbref else "bold"

        seq["run"] = get_run(s)
        seq_extra["dir"] = seq.get("dir", None)
        if s.is_motion_corrected:
            seq["rec"] = "moco"
    elif 'fl3d5' in s.sequence_name:
        seq["label"] = "VFA"
        seq["acq"] = "stage"
        seq["type"] = "anat"
        seq_extra["flip"] = s.custom['flip_angle']
    elif "tfi2d1" in s.sequence_name:
        if "T1 MAP" in image_type:
            seq["label"] = "T1map"
        else:
            seq["label"] = "IRT1"
        if "MOCO" in image_type:
            seq["rec"] = "moco"
        seq["acq"] = "myomaps"
        seq["type"] = "anat"

    ################## SPINAL CORD PROTOCOL #####################
    elif "spcR_100" in s.sequence_name:
        seq["label"] = "T2w"
    #        seq['bp'] = 'spine'
    elif "*me2d1r3" in s.sequence_name:
        seq["label"] = "T2starw"

    ### GE hyperband
    elif "hypermepi" in s.sequence_name:
        seq["type"] = "func"
        seq["label"] = "bold"

    # fix bug with tarred dicoms being indexed in the wrong order, resulting in phase tag
    if seq["label"] == "sbref" and "part" in seq_extra:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        del seq_extra["part"]

    seq_extra["series_id"] = s.series_id
        
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
    skip_derived = False

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
        image_type = s.custom['image_type_text'] or s.image_type
        if (
            skip_derived
            and (s.is_derived or ("MPR" in image_type))
            and not s.is_motion_corrected
            and not "UNI" in image_type
        ):
            skipped.append(s.series_id)
            lgr.info("Ignoring derived data %s", s.series_id)
            continue

        seq_type = bids_info["type"]
        seq_label = bids_info["label"]

        if not s.sequence_name:
            s.sequence_name = s.custom_info['pulse_sequence_name']
            
        if (seq_type == "fmap" and seq_label == "epi" and bids_extra['part']=='phase' and seq_label=='bold'):
            continue

        if ((seq_type == "fmap" and seq_label == "epi") or
            (sbref_as_fieldmap and seq_label == "sbref" and seq_type=='func')
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

    info = dedup_all_bids_extra(info, all_bids_infos)
    info = get_dups_marked(info)  # mark duplicate ones with __dup-0x suffix

    info = dict(
        info
    )  # convert to dict since outside functionality depends on it being a basic dict

    for k, i in info.items():
        lgr.info(f"{k} {i}")

    return info


DEDUP_ENTITIES = [
    ("rec", "value"),
    ("part", "value"),
    ("dir", "value"),
    ("flip", "index"),
    ("series_id", "unique_as_run"),
]


def dedup_all_bids_extra(info, bids_infos):
    # add `rec-` or `part-` to dedup series originating from the same acquisition
    info = info.copy()
    for extra, val in DEDUP_ENTITIES:
        info = dedup_bids_extra(info, bids_infos, extra, val)
    return info

def dedup_bids_extra(info, bids_infos, extra, dedup_val):
    info = info.copy()
    for template, series_ids in list(info.items()):

        if len(series_ids) >= 2:
            lgr.warning("Detected %d run(s) for template %s: %s",
                        len(series_ids), template[0], series_ids)

    
            bids_extra_values = [bids_infos[sid][1].get(extra) for sid in series_ids]
            lgr.info(f'{extra} values {bids_extra_values}')
            if len(set(filter(lambda x: x is None, bids_extra_values))) < 2:
                continue #does not differentiate series

            lgr.info(f"dedup series using {extra}")
            
            for rid, sid in enumerate(list(series_ids)): #need a copy of list because we are removing elements in that loop

                series_bids_info, series_bids_extra = bids_infos[sid]
                
                extra_value = series_bids_extra.get(extra)

                #if series_id are unique, these are not re-run, do not mark as dups
                if dedup_val == "unique_as_run":
                    series_bids_info["run"] = rid+1
                # use index of sorted values (eg. multiple flip angles)
                elif dedup_val == "index":
                    bids_extra_values = sorted(bids_extra_values)
                    extra_value = bids_extra_values.index(extra_value)+1
                    
                new_template = generate_bids_key(
                    series_bids_info["type"],
                    series_bids_info["label"],
                    "",
                    series_bids_info,
                    show_dir=series_bids_info["type"] in ["fmap", "dwi"] or extra=="dir",
                    outtype=("nii.gz",),
                    **{extra: extra_value})

                if new_template not in info:
                    info[new_template] = []
                info[new_template].append(sid)
                info[template].remove(sid)
                if not len(info[template]):
                    del info[template]

    return info
