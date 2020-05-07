import os, re
import nibabel.nicom.dicomwrappers as nb_dw
from heudiconv.heuristics import reproin
from heudiconv.heuristics.reproin import OrderedDict, create_key, get_dups_marked, parse_series_spec, sanitize_str, lgr, series_spec_fields

def infotoids(seqinfos, outdir):

    ex_dcm  = nb_dw.wrapper_from_file(next(seqinfos.__iter__()).example_dcm_file_path)

    pi = str(ex_dcm.dcm_data.ReferringPhysicianName)
    study_name = str(ex_dcm.dcm_data.StudyDescription)
    patient_name = str(ex_dcm.dcm_data.PatientName)

    print(pi, study_name, patient_name)
    study_path = study_name.split('^')

    #example_dcm_file = next(seqinfos.__iter__()).example_dcm_file

    rema = re.match('(([^_]*)_)?(([^_]*)_)?p([0-9]*)_([a-z]*)([0-9]*)', patient_name)

    locator = os.path.join(pi,*study_path)

    study_name = rema.group(1)
    sub_study_name = rema.group(3)
    subject_id = rema.group(5)
    session_type = rema.group(6)
    session_id = rema.group(7)

    return {
        'locator': locator,
        # Sessions to be deduced yet from the names etc TODO
        'session': session_type+session_id,
        'subject': subject_id,
    }

def create_key2(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict2(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    allowed template fields - follow python string module:
    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    # paths done in BIDS format
    t1w = create_key('{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_rec-{rec}_T1w')
    t2w = create_key('{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_rec-{rec}_T2w')
    swi = create_key('{bids_subject_session_dir}/swi/{bids_subject_session_prefix}_swi')
    mp2rage = create_key('{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_inv-{inv}_rec-{rec}_MP2RAGE')
    mp2rage_uni = create_key('{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_rec-{rec}_MP2RAGE')
    mtw = create_key('{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_acq-{acq}_rec-{rec}_MTw')

    dwi = create_key('{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_acq-{acq}_DWI')
    dwi_sbref = create_key('{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_acq-{acq}_DWI_sbref')


    fmri = create_key('{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-{task}_acq-{acq}_run-{run:01d}_bold')
    fmri_sbref = create_key('{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-{task}_acq-{acq}_run-{run:01d}_bold_sbref')

    info = {t1w:[], t2w:[], swi:[], mp2rage:[], mp2rage_uni:[], mtw:[], dwi:[], dwi_sbref:[], fmri:[], fmri_sbref:[]}

    for s in seqinfo:

        ex_dcm  = nb_dw.wrapper_from_file(s.example_dcm_file_path)

        try:
            pedir_pos = bool(ex_dcm.csa_header['tags']['PhaseEncodingDirectionPositive']['items'][0])
        except:
            pedir_pos = None

        # todo: be more specific for T1 and T2
        if (s.dim4 == 1) and ('T1' in s.protocol_name) and ('tfl3d1_16ns' in s.sequence_name) :
            info[t1w].append({'item': s.series_id, 'rec': s.image_type[-1]})
        if (s.dim4 == 1) and ('T2' in s.protocol_name) and ('spc_314ns' in s.sequence_name):
            info[t2w].append({'item': s.series_id, 'rec': s.image_type[-1]})
        if (s.dim4 == 1) and ('swi' in s.protocol_name) and not ('MNIP' in s.image_type):
            info[swi].append({'item': s.series_id})
        if (s.dim4 == 1) and ('mp2rage' in s.protocol_name):
            rec = 'DIS'
            if 'ND' in s.image_type:
                rec = 'ND'

            if 'INV' in s.series_description:
                inv = 1
                if 'INV2' in s.series_description:
                    inv = 2
                info[mp2rage].append({'item': s.series_id, 'inv':inv, 'rec':rec})
            else:
                info[mp2rage_uni].append({'item': s.series_id, 'rec':rec})

        if (s.dim4 == 1) and ('MTw' in s.protocol_name):
            acq = 'Off'
            if 'On' in s.protocol_name:
                acq = 'On'
            rec = 'DIS'
            if 'ND' in s.image_type:
                rec = 'ND'
            info[mtw].append({'item': s.series_id, 'acq':acq, 'rec':rec})

        # Siemens or CMRR diffusion sequence, exclude DERIVED processing at the console
        if (('ep_b0' in s.sequence_name) or ('epse2d1_110' in s.sequence_name)) and not ('DERIVED' in s.image_type):
            pe_dir = 'PA'
            if pedir_pos:
                pe_dir = 'AP'
            if 'Single-band reference' in ex_dcm.dcm_data.ImageComments: # CMRR
                info[dwi_sbref].append({'item': s.series_id, 'dir':pe_dir})
            else:
                info[dwi].append({'item': s.series_id, 'dir':pe_dir})

        # CMRR or Siemens functional sequences
        if ('epfid2d1_104' in s.sequence_name) or ('epfid2d1_96' in s.protocol_name):
            task_name = get_task(s)
            run = get_run(s)
            if not s.is_motion_corrected:
                pe_dir = 'AP'
                if pedir_pos:
                    pe_dir = 'PA'
                if 'Single-band reference' in ex_dcm.dcm_data.ImageComments:
                    info[fmri_sbref].append({'item': s.series_id, 'task':task_name, 'acq':pe_dir, 'run':run})
                else:
                    info[fmri].append({'item': s.series_id, 'task':task_name, 'acq': pe_dir, 'run':run})
    print(info)
    return info


def get_task(s):
    mtch = re.match('.*_task\-([^_]+).*', s.series_id)
    if mtch is not None:
        return mtch.group(1)
    else:
        return None

def get_run(s):
    mtch = re.match('.*run\-([^_]+).*', s.series_id)
    if mtch is not None:
        return mtch.group(1)
    else:
        return 1

rec_exclude = ['ORIGINAL', 'PRIMARY', 'M', 'MB', 'ND', 'MOSAIC','NONE', 'DIFFUSION']

def get_seq_bids_info(s, ex_dcm):

    seq = {
        'type':'anat', # by default to make code concise
        'label':None,
        }
    for it in s.image_type[2:]:
        if it not in rec_exclude:
            seq['rec'] = it.lower()

    try:
        pedir = ex_dcm.dcm_data.InPlanePhaseEncodingDirection
        if 'COL' in pedir:
            pedir = 'AP'
        else:
            pedir = 'LR'
        pedir_pos = bool(ex_dcm.csa_header['tags']['PhaseEncodingDirectionPositive']['items'][0])
        # TODO: get AP/LR/..
        seq['dir'] = pedir if pedir_pos else pedir[::-1]
    except:
        pass

    #label bodypart which are not brain, mainly for spine if we set the dicom fields at the console properly
    bodypart = ex_dcm.dcm_data.get('BodyPartExamined',None)
    if bodypart is not None and bodypart!='BRAIN':
        seq['acq'] = bodypart.lower()

    scan_options = ex_dcm.dcm_data.get('ScanOptions',None)

    # Anats
    if 'localizer' in s.protocol_name:
        seq['label'] = 'localizer'
    elif 'AAHead_Scout' in s.protocol_name:
        seq['label'] = 'scout'
    elif (s.dim4 == 1) and ('T1' in s.protocol_name) and ('tfl3d1_16ns' in s.sequence_name) :
        seq['label'] = 'T1w'
    elif (s.dim4 == 1) and ('T2' in s.protocol_name) and ('spc_314ns' in s.sequence_name):
        seq['label'] = 'T2w'
    elif (s.dim4 == 1) and ('mp2rage' in s.protocol_name) and not ('memp2rage' in s.protocol_name):
        seq['label'] = 'MP2RAGE'
        if 'INV1' in s.protocol_name:
            seq['inv'] = 1
        elif 'INV2' in s.protocol_name:
            seq['inv'] = 2
        elif 'UNI' in s.protocol_name:
            seq['acq'] = 'UNI'

    elif (s.dim4 == 1) and ('MTw' in s.protocol_name):
        seq['label'] = 'MTw'
        seq['acq'] = 'off'
        if 'On' in s.protocol_name:
            seq['acq'] = 'on'

    # SWI
    elif (s.dim4 == 1) and ('swi' in s.protocol_name):
        seq['type'] = 'swi'
        if not ('MNIP' in s.image_type):
            seq['label'] = 'swi'
        else:
            seq['label'] = 'minIP'

    # Siemens or CMRR diffusion sequence, exclude DERIVED (processing at the console)
    elif (('ep_b' in s.sequence_name) or
          ('ez_b' in s.sequence_name) or
          ('epse2d1_110' in s.sequence_name)) and \
        not ('DERIVED' in s.image_type):
        seq['type'] = 'dwi'
        seq['label'] = 'dwi'

    # CMRR or Siemens functional sequences
    elif ('epfid2d1' in s.sequence_name):
        seq['type'] = 'func'
        seq['label'] = 'bold'
        seq['task'] = get_task(s)
        seq['run'] = get_run(s)
        if s.is_motion_corrected:
            seq['rec'] = 'moco'

    ################## SPINAL CORD PROTOCOL #####################
    #elif ('spcR_100'  in s.sequence_name):
    #    seq['label'] = 'GRE'

    # CMRR bold and dwi
    if 'ImageComments' in ex_dcm.dcm_data and 'Single-band reference' in ex_dcm.dcm_data.ImageComments:
        seq['label'] = 'sbref'
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

    info = OrderedDict()
    skipped, skipped_unknown = [], []
    current_run = 0
    run_label = None   # run-
    dcm_image_iod_spec = None
    skip_derived = False

    outtype = ('nii.gz',)
    sbref_as_fieldmap = True # duplicate sbref in fmap dir to be used by topup
    prefix = ''

    fieldmap_runs = {}

    for s in seqinfo:

        ex_dcm  = nb_dw.wrapper_from_file(s.example_dcm_file_path)

        bids_info = get_seq_bids_info(s, ex_dcm)

        # XXX: skip derived sequences, we don't store them to avoid polluting
        # the directory, unless it is the motion corrected ones
        # (will get _rec-moco suffix)
        if skip_derived and s.is_derived and not s.is_motion_corrected:
            skipped.append(s.series_id)
            lgr.debug("Ignoring derived data %s", s.series_id)
            continue

        seq_type = bids_info['type']
        seq_label = bids_info['label']

        suffix_parts = [
            None if not bids_info.get('task') else "task-%s" % bids_info['task'],
            None if not bids_info.get('acq') else "acq-%s" % bids_info['acq'],
            None if not bids_info.get('rec') else "rec-%s" % bids_info['rec'],
            None if not bids_info.get('run') else "run-%02d" % int(bids_info['run']),
            seq_label,
        ]
        # filter tose which are None, and join with _
        suffix = '_'.join(filter(bool, suffix_parts))

        #if "_Scout" in s.series_description or \
        #        (seqtype == 'anat' and seqtype_label and seqtype_label.startswith('scout')):
        #    outtype = ('dicom',)
        #else:
        #    outtype = ('nii.gz', 'dicom')

        template = create_key(seq_type, suffix, prefix=prefix, outtype=outtype)

        # we wanted ordered dict for consistent demarcation of dups
        if template not in info:
            info[template] = []
        info[template].append(s.series_id)

        # duplicate sbref to be used as fieldmap
        if sbref_as_fieldmap and seq_label == 'sbref':
            pe_dir = bids_info.get('dir', None)
            if not pe_dir in fieldmap_runs:
                fieldmap_runs[pe_dir] = 0
            fieldmap_runs[pe_dir] += 1

            run = fieldmap_runs[pe_dir]
            suffix_parts = [
                None if not pe_dir else "dir-%s" % bids_info['dir'],
                "run-%02d" % int(bids_info['run']),
                'epi',
            ]
            suffix = '_'.join(filter(bool, suffix_parts))
            template = create_key('fmap', suffix, prefix=prefix, outtype=outtype)
            if template not in info:
                info[template] = []
            info[template].append(s.series_id)

    if skipped:
        lgr.info("Skipped %d sequences: %s" % (len(skipped), skipped))
    if skipped_unknown:
        lgr.warning("Could not figure out where to stick %d sequences: %s" %
                    (len(skipped_unknown), skipped_unknown))

    info = get_dups_marked(info)  # mark duplicate ones with __dup-0x suffix

    info = dict(info)  # convert to dict since outside functionality depends on it being a basic dict

    return info
