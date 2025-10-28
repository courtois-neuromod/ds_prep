import os, re, glob
from frozendict import frozendict
import nibabel.nicom.dicomwrappers as nb_dw
import nibabel.nicom.ascconv as nb_asconv
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
import pprint

ASCCONV_RE = re.compile(
    r".*### ASCCONV BEGIN((?:\s*[^=\s]+=[^=\s]+)*) ###\n(.*?)\n### ASCCONV END ###.*",
    flags=re.M | re.S,
)


class SeqinfoPrettyPrinter(pprint.PrettyPrinter):
    def _format(self, obj, *args, **kwargs):
        if isinstance(obj, frozendict):
            obj = dict(**obj)
        return super()._format(obj, *args, **kwargs)


pprinter = SeqinfoPrettyPrinter()


### modified from nibabel
def parse_ascconv(ascconv_str, str_delim='"'):
    """Parse the 'ASCCONV' format from `input_str`.

    Parameters
    ----------
    ascconv_str : str
        The string we are parsing
    str_delim : str, optional
        String delimiter.  Typically '"' or '""'

    Returns
    -------
    prot_dict : OrderedDict
        Meta data pulled from the ASCCONV section.
    attrs : OrderedDict
        Any attributes stored in the 'ASCCONV BEGIN' line

    Raises
    ------
    AsconvParseError
        A line of the ASCCONV section could not be parsed.
    """
    attrs, content = ASCCONV_RE.match(ascconv_str).groups()
    attrs = OrderedDict(tuple(x.split("=")) for x in attrs.split())
    # Normalize string start / end markers to something Python understands
    content = content.replace(str_delim, '"').replace("\\", "\\\\")
    ascconv = {}

    for l in content.split("\n"):
        k, v = l.split("=")
        v = v.strip()
        if '"' in v or not re.match("^[0-9.]+$", v):
            v = v.replace('"', "")
        elif "." in v:
            v = float(v)
        else:
            v = int(v)
        ascconv[k.strip()] = v
    return ascconv, attrs


SIEMENS_TAGS = {
    "slice_orient": (0x0051, 0x100E),
    "ReceiveCoilName": (0x0051, 0x100F),
}

XA_SDI_TAGS = {
    "image_type_text": (0x0021, 0x1175),
    "InPlanePhaseEncodingDirection_pos": (0x0021, 0x111C),
    "ice_dims": (0x0021, 0x1106),
    "image_history": (0x0021, 0x1176),
}
XA_SDI2_TAGS = {
    "InPlanePhaseEncodingDirection_pos": (
        0x0021,
        0x101C,
    ),  # damn siemens changing their format every morning
}


def dicom_value_to_python(tag):
    if tag:
        val = tag.value if isinstance(tag, pydicom.dataelem.DataElement) else tag
        if isinstance(val, pydicom.multival.MultiValue):
            return tuple(val)
        elif isinstance(val, pydicom.valuerep.IS):
            return int(val)
        elif isinstance(val, bytes):
            return val.decode("utf-8")
        elif isinstance(val, str):
            return val
        else:
            return val


ENHANCED_MR_SEQUENCES = [
    "MRFOVGeometrySequence",
    "MRReceiveCoilSequence",
    "MRTimingAndRelatedParametersSequence",
    "MRModifierSequence",
    "MRImagingModifierSequence",
]


def custom_seqinfo(wrapper, series_files):
    # print('calling custom_seqinfo', wrapper, series_files)

    image_history = ice_dims = pedir_pos = image_type_text = None
    InPlanePhaseEncodingDirection = wrapper.dcm_data.get(
        "InPlanePhaseEncodingDirection", None
    )
    slice_orient = wrapper.dcm_data.get([0x0051, 0x100E])
    ReceiveCoilName = wrapper.dcm_data.get((0x0051, 0x100F))
    ReceiveCoilName = ReceiveCoilName.value if ReceiveCoilName else None
    custom_info = {}
    SequenceVariant = (
        dicom_value_to_python(wrapper.dcm_data.get("SequenceVariant"))
        if "SequenceVariant" in wrapper.dcm_data
        else None
    )
    FlipAngle = wrapper.dcm_data.FlipAngle if "FlipAngle" in wrapper.dcm_data else None
    scan_options = (str(wrapper.dcm_data.get("ScanOptions", None)),)
    series_has_diff = None
    InPlanePhaseEncodingDirection_pos = None
    seq_fname = None
    mrprot = None
    n_inversion_contrasts = None
    b_values = None
    wip_mem_blocks_alfree = None

    enh_custom_tags = {}

    if hasattr(wrapper, "csa_header"):
        InPlanePhaseEncodingDirection_pos = wrapper.csa_header["tags"][
            "PhaseEncodingDirectionPositive"
        ]["items"]
        InPlanePhaseEncodingDirection_pos = (
            InPlanePhaseEncodingDirection_pos[0]
            if len(InPlanePhaseEncodingDirection_pos) > 0
            else None
        )
        image_history = wrapper.csa_header["tags"]["ImageHistory"]["items"]
        ice_dims = wrapper.csa_header["tags"]["ICE_Dims"]["items"][0]
        try:
            csa_series = nb_dw.csar.get_csa_header(wrapper.dcm_data, "series")
            mrprot = csa_series["tags"]["MrPhoenixProtocol"]["items"][0]
        except Exception as e:
            lgr.error("failed to read MrPhoenixProtocol", e)

    elif hasattr(wrapper, "frames"):
        # XA multiframe enhanced dicoms
        for section, tags in zip([0x11FE, 0x10FE], [XA_SDI_TAGS, XA_SDI2_TAGS]):
            if (0x0021, section) in wrapper.frames[0]:
                frame0_sdi_private_info = wrapper.frames[0][(0x0021, section)][0]
                for key, code in tags.items():
                    if code in frame0_sdi_private_info and not custom_info.get(key):
                        tag = frame0_sdi_private_info.get(code)
                        custom_info[key] = dicom_value_to_python(tag)

        for enh_mr_seq in ENHANCED_MR_SEQUENCES:
            if enh_mr_seq in wrapper.shared:
                seq = wrapper.shared.get(enh_mr_seq)[0]
                for tag in seq.dir():
                    val = getattr(seq, tag)
                    if not isinstance(val, pydicom.sequence.Sequence):
                        enh_custom_tags[tag] = getattr(seq, tag)

        if (0x0021, 0x10FE) in wrapper.shared:
            sds1 = wrapper.shared[(0x0021, 0x10FE)][0]
            scan_options = dicom_value_to_python(sds1.get((0x0021, 0x105C)))
            mrprot = dicom_value_to_python(sds1.get((0x0021, 0x1019)))

    if mrprot is not None:
        try:
            ascconv, _ = parse_ascconv(mrprot, '""')
            b_values = tuple(
                filter(
                    bool, [ascconv.get(f"sDiffusion.alBValue[{i}]") for i in range(10)]
                )
            )
            series_has_diff = any(b_values)
            seq_fname = ascconv.get("tSequenceFileName")
            n_inversion_contrasts = int(ascconv.get("lInvContrasts"))
            wip_mem_blocks_alfree = tuple(
                [ascconv.get(f"sWipMemBlock.alFree[{i}]", None) for i in range(10)]
            )

        except Exception as e:
            lgr.error("failed to read/parse series CSA ascconv", e)

    ge_userdata, ge_acq_bits = read_ge_userdata(wrapper)
    if ge_acq_bits:
        InPlanePhaseEncodingDirection_pos = (ge_acq_bits & 4) > 0
        custom_info["nframes"] = getattr(ge_userdata, "rhr_rh_nframes")
    if (0x0043, 0x10B3) in wrapper.dcm_data:
        custom_info["ge_eddy"] = sum(wrapper.dcm_data.get((0x0043, 0x10B3)).value) > 1

    custom_info2 = {
        "patient_name": wrapper.dcm_data.PatientName,
        "InPlanePhaseEncodingDirection": InPlanePhaseEncodingDirection,
        "InPlanePhaseEncodingDirection_pos": InPlanePhaseEncodingDirection_pos,
        "pulse_sequence_name": wrapper.dcm_data.get("PulseSequenceName", None),
        "body_part": wrapper.dcm_data.get("BodyPartExamined", None),
        "scan_options": scan_options,
        "image_comments": wrapper.dcm_data.get("ImageComments", ""),
        "slice_orient": str(slice_orient.value) if slice_orient else None,
        "echo_number": dicom_value_to_python(wrapper.dcm_data.get("EchoNumber", None)),
        "FlipAngle": FlipAngle,
        "rescale_slope": wrapper.dcm_data.get("RescaleSlope", None),
        "ReceiveCoilName": ReceiveCoilName,
        "SequenceVariant": SequenceVariant,
        "image_history": (
            ";".join(filter(len, image_history)) if image_history else None
        ),
        "ice_dims": ice_dims,
        "image_type_text": image_type_text,
        "series_has_diff": series_has_diff,
        "b_values": b_values,
        "seq_fname": seq_fname,
        "n_inversion_contrasts": n_inversion_contrasts,
        "wip_mem_blocks_alfree": wip_mem_blocks_alfree,
    }

    custom_info = custom_info2 | custom_info | enh_custom_tags

    custom_info = frozendict(custom_info)

    return custom_info


GE_USERDATA_TAG = (0x0043, 0x102A)


def read_ge_userdata(wrapper):
    import spec2nii.GE.ge_read_pfile
    import io, struct

    if GE_USERDATA_TAG in wrapper.dcm_data:
        user_data = wrapper.dcm_data[GE_USERDATA_TAG].value

        try:
            f = io.BytesIO()
            f.write(user_data)
            f.seek(24)
            (hdr_offset,) = struct.unpack("i", f.read(4))
            f.seek(hdr_offset)
            (version,) = struct.unpack("f", f.read(4))

            f.seek(hdr_offset + 0x0030 + 0x004C)
            acq_bits = struct.unpack("i", f.read(4))[0]

            class PfileHeaderLittle(spec2nii.GE.ge_read_pfile.ct.LittleEndianStructure):
                """
                Contains the ctypes Structure for a GE P-file rdb header.
                Dynamically allocate the ctypes _fields_ list later depending on revision
                """

                _pack_ = 1
                _fields_ = spec2nii.GE.ge_read_pfile.get_pfile_hdr_fields(version)

            hdr = PfileHeaderLittle()
            f.seek(hdr_offset)
            f.readinto(hdr)
            f.close()
            return hdr, acq_bits
        except Exception as e:
            lgr.warning("parsing of GE userdata failed", e)
    return None, None


INFOTOIDS_REGEXS = [
    # neuromod pattern
    "p(?P<subject>[0-9]*)_(?P<study>[a-zA-Z0-9]*)(?P<session>[0-9]{3})",
    # generic patterns
    "(pilot|dev)_(?P<study>[^_]*)_(?P<subject>[a-zA-Z0-9]+)(?:_(?P<session>[a-zA-Z0-9]+))?",
    "(?P<study>[^_]*)_(?P<subject>[a-zA-Z0-9]+)(?:_(?P<session>[a-zA-Z0-9]+))?",
    "(?P<study>[^_]*) (?P<subject>[a-zA-Z0-9]+)(?: (?P<session>[a-zA-Z0-9]+))?",
    "(?P<study>[^_]*)_(?P<subject>[a-zA-Z0-9]+)(?:_(?P<session>[a-zA-Z0-9]+))?.*",
    "^(?P<study>[^_]*)-(?P<subject>[a-zA-Z0-9]+)(?:-(?P<session>[a-zA-Z0-9]+))$",
    "^(?P<study>[^_]*)-(?P<subject>[a-zA-Z0-9]+)(?:-(?P<session>[a-zA-Z0-9]+))?$"
    "(?P<study>[^_]*)_sub-(?P<subject>[a-zA-Z0-9]+)(?:_ses-(?P<session>[a-zA-Z0-9]+))?",
]


def infotoids(seqinfos, outdir):
    seqinfo = next(seqinfos.__iter__())

    pi = str(seqinfo.referring_physician_name)
    study_name = str(seqinfo.study_description)
    patient_name = str(seqinfo.custom["patient_name"])

    study_path = study_name.split("^")
    locator = os.path.join(pi, *study_path)

    for itoids_regex in INFOTOIDS_REGEXS:
        rema = re.match(itoids_regex, patient_name)
        if rema:
            res = rema.groupdict()
            lgr.info(f"infotoids match: {res}")
            return res
    return {"subject": "unknown"}


def get_task(s):
    mtch = re.match(".*[-_]task-([^_]+).*", s.series_id)
    if mtch is None:
        mtch = re.match(".*-task_([^_]+).*", s.series_id)  # for floc messup
    if mtch is not None:
        task = mtch.group(1).split("-")
        if len(task) > 1:
            return task[1]
        return task[0]
    else:
        lgr.error("could not detect task from series description: assuming rest")
        return "rest"


def get_run(s):
    mtch = re.match(".*run-([^_]+).*", s.series_id)
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
    "OTHER",
    "MOSAIC",
    "NONE",
    "DIFFUSION",
    "UNI",
] + [f"TE{i}" for i in range(9)]


def get_seq_bids_info(s, stage_as_megre=True):
    seq = {
        "type": "anat",  # by default to make code concise
        "label": None,
    }

    image_type = s.custom["image_type_text"] or s.image_type

    # skip derived, but can be overriden by specific sequences (eg MP2RAGE)
    # keep moco series
    skip = (s.is_derived and not s.is_motion_corrected) or any(
        it in image_type for it in ["MPR", "CSA REPORT", "SPECTROSCOPY", "PHYSIO"]
    )
    if skip:
        return {}, {}, skip

    seq_extra = {}
    it_rec = ""
    for it in image_type[2:]:
        if it not in rec_exclude:
            it_rec += it.lower().capitalize()
    if it_rec:
        seq_extra["rec"] = it_rec

    seq_extra["part"] = (
        "mag" if "M" in image_type else ("phase" if "P" in image_type else None)
    )

    if (
        s.custom.get("InPlanePhaseEncodingDirection") is not None
        and s.custom.get("InPlanePhaseEncodingDirection_pos") is not None
    ):
        pedir = s.custom["InPlanePhaseEncodingDirection"][:3]
        if "COL" == pedir:
            pedir = "AP"
        elif "ROW" == pedir:
            pedir = "LR"
        else:
            pedir = None

        if pedir is not None:
            pedir_pos = bool(int(s.custom["InPlanePhaseEncodingDirection_pos"]))
            seq_extra["dir"] = pedir if pedir_pos else pedir[::-1]

    # label bodypart which are not brain, mainly for spine if we set the dicom fields at the console properly
    bodypart = s.custom["body_part"]  # ex_dcm.dcm_data.get("BodyPartExamined", None)
    if bodypart is not None and bodypart not in ["BRAIN", "HEAD"]:
        seq["bp"] = bodypart.lower()

    scan_options = s.custom["scan_options"]
    image_comments = s.custom["image_comments"]

    # CMRR bold, dwi, spinecho
    is_sbref = "Single-band reference" in image_comments
    if "cmrr" in s.custom["seq_fname"]:
        seq_extra["multiband"] = "MB" if "MB" in image_type else "SB"

    if s.custom["ice_dims"] and s.custom["ice_dims"][0] != "X" and not s.is_derived:
        seq["rec"] = "uncombined"
    # Anats
    if "localizer" in s.protocol_name.lower():
        seq["label"] = "localizer"
        slice_orient = s.custom["slice_orient"]
        if slice_orient is not None:
            seq_extra["acq"] = slice_orient.lower()
        skip = True  # skip for now, until https://github.com/nipy/heudiconv/pull/788 is resolved
    elif s.custom["seq_fname"] and "AALScout" in s.custom["seq_fname"]:
        seq["label"] = "scout"
    elif (
        (s.dim4 == 1)
        and s.custom.get("n_inversion_contrasts") == 1
        and ("tfl3d1" in s.sequence_name and "ns" in s.sequence_name)
    ):
        seq["label"] = "T1w"
    elif (
        (s.dim4 == 1) and ("T2" in s.protocol_name) and ("spc_314ns" in s.sequence_name)
    ):
        seq["label"] = "T2w"
    elif (
        ("*tfl3d1" in s.sequence_name)
        and s.custom.get("n_inversion_contrasts") == 2
        and not ("memp2rage" in s.protocol_name)
    ):
        seq["acq"] = "mp2rage"
        seq["label"] = "MP2RAGE"
        if "INV1" in s.series_description:
            seq["inv"] = 1
        elif "INV2" in s.series_description:
            seq["inv"] = 2
        elif "UNI" in image_type:
            seq["label"] = "UNIT1"
        elif "T1_MAP" in image_type or "T1 MAP" in image_type:
            seq["label"] = "T1map"
        skip = False
    elif "spcir" in s.sequence_name or "tir" in s.sequence_name:
        seq["label"] = "FLAIR"

    # GRE acquisition
    elif s.sequence_name == "*fl3d1":
        seq["label"] = "MTS"
        seq["mt"] = "on" if ("MT" in scan_options) else "off"
        seq_extra["flip"] = s.custom.get("FlipAngle")
    elif "tfl2d1" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "TB1TFL"
        seq["acq"] = "famp" if "flip angle map" in image_comments else "anat"
    elif "epse2d1_9" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "TB1EPI"
        seq["flip"] = 1 if s.custom.get("FlipAngle") < 90 else 2

    elif "fm2d2r" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = (
            "phasediff"
            if "PHASE" in image_type or "P" in image_type
            else ("magnitude%d" % (s.custom.get("echo_number") or 1))
        )
    # memprage from MGH
    elif "tfl_me" in s.sequence_name:
        seq["label"] = "T1w"
        seq["acq"] = "memprage"

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
        if s.custom["series_has_diff"] == False:
            # if the series has no diffusion direction, all b0 then export it as fmap
            # dcm2niix will not produce bvals/bvecs and it won't be valid DWI
            seq["type"] = "fmap"
            seq["label"] = "epi"
            seq["dir"] = seq_extra["dir"]
            seq_extra["acq"] = (
                "dwi" if "diff" in s.custom.get("seq_fname") else "spinecho"
            )

        else:
            seq["type"] = "dwi"
            seq["label"] = "sbref" if is_sbref else "dwi"
            if s.custom.get("b_values"):
                seq_extra["acq"] = "".join(
                    [f"b{bval}" for bval in s.custom.get("b_values")]
                )
        seq_extra["series_description"] = s.series_description
        # dumb far-fetched heuristics, no info in dicoms see https://github.com/CMRR-C2P/MB/issues/305
        seq_extra["part"] = "phase" if s.custom["rescale_slope"] else "mag"

    # CMRR or Siemens functional sequences
    elif "epfid2d" in s.sequence_name:

        # if no task, this is a fieldmap
        #        if "AP" in s.series_id and not seq["task"]:
        #            seq["type"] = "fmap"
        #            seq["label"] = "epi"
        #            seq["acq"] = "sbref" if is_sbref else "bold"
        #        else:
        if "fmap" in s.series_description:
            seq["type"] = "fmap"
            seq["label"] = "epi"
            seq["dir"] = seq_extra["dir"]
        else:
            seq["task"] = get_task(s)
            seq["type"] = "func"
            seq["label"] = "sbref" if is_sbref else "bold"

        seq["run"] = get_run(s)
        if s.is_motion_corrected:
            seq["rec"] = "moco"
    elif "fl2d1" in s.sequence_name:
        seq["label"] = "PDw"  # TODO: need to confirm, depends on params
        seq["acq"] = "flash"
    elif "fl3d5" in s.sequence_name:
        seq["acq"] = "stage"
        if stage_as_megre:
            seq["label"] = "MEGRE"
            seq["acq"] += f"fa{int(s.custom['FlipAngle'])}"
        else:  # VFA
            seq["label"] = "VFA"
            seq_extra["flip"] = s.custom["FlipAngle"]
    elif "tfi2d1" in s.sequence_name:
        if "T1 MAP" in image_type:
            seq["label"] = "T1map"
        else:
            seq["label"] = "IRT1"
        if "MOCO" in image_type:
            seq["rec"] = "moco"
        seq["acq"] = "myomaps"

    ################## SPINAL CORD PROTOCOL #####################
    elif "spcR_100" in s.sequence_name:
        seq["label"] = "T2w"
    #        seq['bp'] = 'spine'
    elif "*me2d1r3" in s.sequence_name:
        seq["label"] = "T2starw"
        seq["type"] = "fmap"
        seq["label"] = "B1Map"

    # iHMT_rage for GE and Siemens (Marseille)
    elif "ihMTRAGE_v1" in s.sequence_name or "ihMT_tfl3d1_16" in s.sequence_name:
        seq["label"] = "MT"
        seq["acq"] = "ihmtrage"
    # ihMT from MNI (Leppert/Tardif)
    elif s.sequence_name.startswith("tfl3d1") and "mni_tfl_MTboost" in s.custom.get(
        "seq_fname", ""
    ):
        seq["mt"] = "off" if s.custom.get("MagnetizationTransfer") == "NONE" else "on"
        pstve = int(s.custom.get("wip_mem_blocks_alfree", [])[4]) > 0
        dual = int(s.custom.get("wip_mem_blocks_alfree", [])[1]) > 1
        seq["label"] = "MTR"
        seq["acq"] = "ihMT"
        if seq["mt"] == "on":
            seq["acq"] = "ihMT" + (
                "dual" if dual else ("single" + "pos" if pstve else "neg")
            )
    # TSE PDT2 series resulting in 2 separate images with varying echo
    if "tse2d2" in s.sequence_name:
        # seq["acq"] = "PDT2"
        seq["label"] = "PDT2"
        # same series, so we cannot sort yet on echo_number
        # seq["label"] = "PDw" if s.custom["echo_number"] == 1 else "T2w"

    ############## GE ####################
    ### GE hyperband
    elif "hypermepi" in s.sequence_name:
        seq["type"] = "func"
        seq["label"] = "bold"
    elif "BRAVO" in s.sequence_name:
        seq["label"] = "T1w"
        if "FILTERED_GEMS" in image_type:
            seq["rec"] = "filtered"
    elif "epi2" in s.sequence_name:
        seq["type"] = seq["label"] = "dwi"
        seq_extra["rec"] = "eddy" if s.custom.get("ge_eddy") else "raw"
    elif "ssfse" in s.sequence_name:
        seq["type"] = "anat"
        seq["label"] = "localizer"
        skip = True
    elif "SWAN" in s.sequence_name:
        seq["label"] = "MEGRE"
        seq["acq"] = f"fa{int(s.custom['FlipAngle'])}"
    elif "B1MAP" in s.sequence_name:
        seq["type"] = "fmap"
        seq["label"] = "TB1map"
    elif "eCSI_rst" in s.sequence_name:
        seq["type"] = "mrs"
        seq["label"] = "mrsi"
        if (
            s.custom.get("nframes") == 64
        ):  # bad heuristics, no MR meta indicate water ref
            seq["label"] = "mrsref"
        skip = True  # skip for now, heudiconv crash w. missing EchoTime in 1 dicom

    if seq["type"] == "func":
        seq["task"] = get_task(s)

    # fix bug with tarred dicoms being indexed in the wrong order, resulting in phase tag
    if seq["label"] == "sbref" and "part" in seq_extra:
        del seq_extra["part"]

    seq_extra["series_id"] = s.series_id
    seq_extra["sequence_name"] = s.sequence_name
    if seq_fname := s.custom.get("seq_fname"):
        seq_extra["sequence_filename"] = seq_fname.split("\\")[-1].replace("_", "")

    # rec not allowed for fmap, use acq instead
    if seq["type"] == "fmap":
        seq.pop("rec", None)
        """
        if seq_extra.get("rec", None):
            if seq_extra.get("acq", None):
                seq_extra["acq"] += seq_extra.pop("rec").capitalize()
            else:
                seq_extra["acq"] = seq_extra.pop("rec")"""

    lgr.debug(f"{seq}, {seq_extra}")
    return seq, seq_extra, skip


def generate_bids_key(prefix, bids_info, show_dir=False, outtype=("nii.gz",)):
    suffix_parts = [
        None if not bids_info.get("task") else "task-%s" % bids_info["task"],
        None if not bids_info.get("acq") else "acq-%s" % bids_info["acq"],
        None if not bids_info.get("ce") else "ce-%s" % bids_info["ce"],
        (
            None
            if not (bids_info.get("dir") and show_dir)
            else "dir-%s" % bids_info["dir"]
        ),
        None if not bids_info.get("rec") else "rec-%s" % bids_info["rec"],
        None if not bids_info.get("tsl") else "tsl-%d" % bids_info["tsl"],
        None if not bids_info.get("loc") else "loc-%s" % bids_info["loc"],
        None if not bids_info.get("bp") else "bp-%s" % bids_info["bp"],
        None if not bids_info.get("run") else "run-%02d" % int(bids_info["run"]),
        None if not bids_info.get("echo") else "echo-%d" % int(bids_info["echo"]),
        None if not bids_info.get("flip") else "flip-%d" % int(bids_info["flip"]),
        None if not bids_info.get("inv") else "inv-%d" % bids_info["inv"],
        None if not bids_info.get("mt") else "mt-%s" % bids_info["mt"],
        None if not bids_info.get("part") else "part-%s" % bids_info["part"],
        bids_info.get("label"),
    ]
    # filter those which are None, and join with _
    suffix = "_".join(filter(bool, suffix_parts))

    return create_key(bids_info["type"], suffix, prefix=prefix, outtype=outtype)


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

    info = OrderedDict()
    skipped, skipped_unknown = [], []
    current_run = 0
    run_label = None  # run-
    dcm_image_iod_spec = None
    skip_derived = True

    outtype = ("nii.gz",)
    # sbref_as_fieldmap = True  # duplicate sbref in fmap dir to be used by topup
    sbref_as_fieldmap = (
        False  # sbref as fieldmaps is still required to use fMRIPrep LTS.
    )
    prefix = ""

    fieldmap_runs = {}
    all_bids_infos = {}

    for s in seqinfo:
        lgr.info(pprinter.pformat(s._asdict()))

        bids_info, bids_extra, skip = get_seq_bids_info(s)
        all_bids_infos[s.series_id] = (bids_info, bids_extra)

        image_type = s.custom["image_type_text"] or s.image_type
        #
        if skip:
            skipped.append(s.series_id)
            lgr.info("Ignoring derived data %s", s.series_id)
            continue

        seq_type = bids_info["type"]
        seq_label = bids_info["label"]

        if (
            (seq_type == "fmap" and seq_label == "epi")
            or (sbref_as_fieldmap and seq_label == "sbref" and seq_type == "func")
        ) and bids_info.get("part") in ["mag", None]:
            InPlanePhaseEncodingDirection = bids_info.get("dir", None)
            if not InPlanePhaseEncodingDirection in fieldmap_runs:
                fieldmap_runs[InPlanePhaseEncodingDirection] = 0
            fieldmap_runs[InPlanePhaseEncodingDirection] += 1
            # override the run number
            run_id = fieldmap_runs[InPlanePhaseEncodingDirection]

            # duplicate sbref to be used as fieldmap
            if sbref_as_fieldmap and seq_label == "sbref":
                suffix_parts = [
                    "acq-sbref",
                    None if not bids_info.get("ce") else "ce-%s" % bids_info["ce"],
                    (
                        None
                        if not InPlanePhaseEncodingDirection
                        else "dir-%s" % bids_info["dir"]
                    ),
                    "run-%02d" % run_id,
                    "epi",
                ]
                suffix = "_".join(filter(bool, suffix_parts))
                template = create_key("fmap", suffix, prefix=prefix, outtype=outtype)
                if template not in info:
                    info[template] = []
                info[template].append(s.series_id)

        show_dir = seq_type in ["fmap", "dwi"] and not seq_label == "TB1TFL"

        template = generate_bids_key(prefix, bids_info, show_dir, outtype)

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

    lgr.info("final mapping:")
    for k, i in info.items():
        lgr.info(f"{k[0]}: {i}")

    return info


DEDUP_ENTITIES = [
    ("part", "value"),
    ("dir", "value"),
    ("acq", "value"),
    ("rec", "value"),
    ("rec", "append:acq"),
    ("flip", "index"),
    ("sequence_filename", "acq"),
    ("multiband", "append:acq"),
    ("series_description", "unique_as_run"),
    ("sequence_name", "unique_as_run"),
]


def dedup_all_bids_extra(info, bids_infos):
    # iteratively add DEDUP_ENTITIES to deduplicate series
    for extra, val in DEDUP_ENTITIES:
        info, bids_info = dedup_bids_extra(info, bids_infos, extra, val)
    return info


def dedup_bids_extra(info, bids_infos, extra, dedup_val):
    info = info.copy()
    for template, series_ids in list(info.items()):
        if len(series_ids) >= 2:
            lgr.warning(
                "Detected %d run(s) for template %s: %s",
                len(series_ids),
                template[0],
                series_ids,
            )

            bids_extra_values = [bids_infos[sid][1].get(extra) for sid in series_ids]
            lgr.info(f"{extra} values {bids_extra_values}")
            if len(set(filter(lambda x: x is not None, bids_extra_values))) < 2:
                continue  # does not differentiate series

            lgr.info(f"dedup series using {extra}")

            target_entity = extra
            for rid, sid in enumerate(
                list(series_ids)
            ):  # need a copy of list because we are removing elements in that loop
                series_bids_info, series_bids_extra = bids_infos[sid]
                extra_value = series_bids_extra.get(extra)

                # if series_description are unique, these are not re-run, do not mark as dups
                if dedup_val == "unique_as_run":
                    series_bids_info["run"] = rid + 1
                # use index of sorted values (eg. multiple flip angles)
                elif dedup_val == "index":
                    bids_extra_values = sorted(bids_extra_values)
                    extra_value = bids_extra_values.index(extra_value) + 1
                elif "append" in dedup_val:
                    target_entity = dedup_val.split(":")[1]
                    extra_value = bids_infos.get(target_entity, "") + extra_value
                # for the info to be kept in chained dedup
                series_bids_info[target_entity] = extra_value

                new_template = generate_bids_key(
                    "",
                    series_bids_info,
                    show_dir=series_bids_info["type"]
                    in ["fmap", "dwi", "func", "perf"],
                    outtype=("nii.gz",),
                )

                if new_template not in info:
                    info[new_template] = []
                info[new_template].append(sid)
                info[template].remove(sid)
                if not len(info[template]):
                    del info[template]

    return info, bids_infos


def filter_dicom(dcm_data):
    ## ignore XA6*A localizer see https://github.com/nipy/nibabel/issues/1392
    if "syngo MR XA6" in str(dcm_data.get("SoftwareVersions")) and (
        "localizer" in dcm_data.get("ProtocolName")
        or dcm_data.get("SeriesDescription") == "MoCoSeries"
    ):
        return True
    return False
