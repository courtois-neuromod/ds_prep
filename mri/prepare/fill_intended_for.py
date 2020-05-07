import sys, os
import shutil, stat
from bids import BIDSLayout
import json
import logging

def fill_intended_for(path):
    path = os.path.abspath(path)
    layout = BIDSLayout(path, validate=False)
    bolds = layout.get(suffix='bold',extensions='.nii.gz')
    json_to_modify = dict()

    for bold in bolds:
        fmaps = layout.get(
            suffix='epi', extension='.nii.gz',
            subject=bold.entities['subject'], session=bold.entities['session'])

        print(bold.path)
        shim_settings = bold.tags['ShimSetting'].value
        # First: get epi fieldmaps with similar ShimSetting
        #print(shim_settings)
        #print([(fm.tags['ShimSetting'].value,fm.tags['PhaseEncodingDirection'].value) for fm in fmaps])
        fmaps_match = [fm for fm in fmaps \
            if fm.tags['ShimSetting'].value == shim_settings]
        # Second: if not 2 fmap found we extend our search
        pedirs = set([fm.tags['PhaseEncodingDirection'].value for fm in fmaps_match])
        print(len(fmaps_match), pedirs)

        # Second: if not 2 fmap found we extend our search
        if len(fmaps_match)<2 or len(pedirs)<2:
            logging.warning("We couldn't find two epi fieldmaps with matching ShimSettings and two pedirs: "\
                + " including other based on ImageOrientationPatient/ImagePositionPatient.")
            fmaps_match.extend([fm for fm in fmaps \
                if fm.tags['global'].value['const']['ImageOrientationPatient'] == bold.tags['global'].value['const']['ImageOrientationPatient'] \
                and fm.tags['global'].value['const']['ImagePositionPatient'] == bold.tags['global'].value['const']['ImagePositionPatient']])

            pedirs = set([fm.tags['PhaseEncodingDirection'].value for fm in fmaps_match])
            print(len(fmaps_match), pedirs)

        # get all fmap possible
        if len(fmaps_match)<2 or len(pedirs)<2:
            logging.warning("We couldn't find two epi fieldmaps with matching ImageOrientationPatient and ImagePositionPatient and two pedirs: "\
                + " including non-matching ones.")
            # TODO: maybe match on time distance
            fmaps_match = fmaps

        # only get 2 images with opposed pedir
        fmaps_match_pe_pos = [fm for fm in fmaps_match if '-' not in fm.tags['PhaseEncodingDirection'].value]
        fmaps_match_pe_pos = fmaps_match_pe_pos[0] if len(fmaps_match_pe_pos) else None
        fmaps_match_pe_neg = [fm for fm in fmaps_match if '-' in fm.tags['PhaseEncodingDirection'].value]
        fmaps_match_pe_neg = fmaps_match_pe_neg[0] if len(fmaps_match_pe_neg) else None

        if not fmaps_match_pe_pos or not fmaps_match_pe_neg:
            logging.error("no matching fieldmaps")
            continue
        for fmap in [fmaps_match_pe_pos, fmaps_match_pe_neg]:
            if ('IntendedFor' not in fmap.tags) or \
                (bold.path not in fmap.tags.get('IntendedFor').value):
                print('adding to IntendedFor')
                fmap_json_path = fmap.get_associations()[0].path
                if fmap_json_path not in json_to_modify:
                    json_to_modify[fmap_json_path] = []
                json_to_modify[fmap_json_path].append(os.path.relpath(bold.path,path))

    print(json_to_modify)
    for json_path, intendedfor in json_to_modify.items():
        logging.info("updating %s"%json_path)
        json_path = os.path.join(path, json_path)
        with open(json_path, 'r', encoding='utf-8') as fd:
            meta = json.load(fd)
        if 'IntendedFor' not in meta:
            meta['IntendedFor'] = []
        meta['IntendedFor'].extend(intendedfor)
        meta['IntendedFor'] = list(set(meta['IntendedFor']))

        #backup_path = json_path + '.bak'
        #if not os.path.exists(backup_path):
        #    shutil.copyfile(json_path, backup_path)

        file_mask = os.stat(json_path)[stat.ST_MODE]
        os.chmod(json_path, file_mask | stat.S_IWUSR)
        with open(json_path, 'w', encoding='utf-8') as fd:
            meta = json.dump(meta, fd, indent=3, sort_keys=True)
        os.chmod(json_path, file_mask)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fill_intended_for(sys.argv[1])
