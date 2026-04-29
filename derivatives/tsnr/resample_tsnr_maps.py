import logging
from pathlib import Path

import bids
import click

from ants import image_read, image_write
from ants.registration import apply_transforms

@click.command()
@click.argument('ds_name', type=str)
@click.argument('tsnr_path', type=click.Path())
@click.argument('fmriprep_path', type=click.Path())
@click.option('--smriprep_path', type=click.Path(), default=None)
@click.option('--to_t1w', is_flag=True)
@click.option('--to_mni', is_flag=True)
def main(ds_name, tsnr_path, fmriprep_path, smriprep_path=None, to_t1w=True, to_mni=False):
    if to_t1w:
        resample_to_T1w(ds_name, tsnr_path, fmriprep_path)
    if to_mni:
        resample_to_MNI152(ds_name, tsnr_path, fmriprep_path, smriprep_path)


def resample_to_T1w(ds_name, tsnr_path, fmriprep_path):
    logger = logging.getLogger(__name__)
    logger.info(f'generating tsnr maps from {Path(tsnr_path).name.split('.')[-1]} data for the {ds_name} dataset')
    logger.info(f"loading tSNR data: {tsnr_path}")
    layout_tsnr = bids.BIDSLayout(tsnr_path, validate=False, is_derivative=True)

    logger.info(f"loading fMRIPrep data: {fmriprep_path}")
    layout_fmriprep = bids.BIDSLayout(fmriprep_path, validate=False, is_derivative=True)
    
    tsnr_maps = [f for f in layout_tsnr.get(suffix='statmap', extension='.nii.gz') if 'stat-tsnr' in f.filename]

    for tsnr_map in tsnr_maps:
        entities = tsnr_map.get_entities()
        entities.update({
            'space': 'T1w',
            'stat': 'tsnr',
        })

        pattern = pattern="sub-{subject}/ses-{session}/{datatype}/sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}[_echo-{echo}]_stat-{stat}_desc-{desc}_{suffix}.nii.gz"
        out_file = layout_tsnr.build_path(entities, pattern, validate=False)

        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(out_file).exists():
            # Get reference images from fMRIPrep directory
            t1w = layout_fmriprep.get(
                subject=entities['subject'],
                session=entities['session'],
                run=entities['run'],
                space='T1w',
                suffix='boldref',
                extension='nii.gz'
            )
            # Validate input
            if len(t1w) == 0:
                raise IOError(f"No T1w boldref file associated with {tsnr_map.filename}")
            elif len(t1w) > 1:
                raise IOError(f"More than one T1w boldref file associated with {tsnr_map.filename}")
            else:
                t1w = t1w[0]
            
            # Get transform from fMRIPrep directory
            transform = layout_fmriprep.get(
                subject=entities['subject'],
                session=entities['session'],
                run=entities['run'],
                desc='coreg',
                suffix='xfm',
                extension='txt'
            )
            # Validate input
            if len(transform) == 0:
                raise IOError(f"No transform file associated with {tsnr_map.filename}")
            elif len(transform) > 1:
                raise IOError(f"More than one transform file associated with {tsnr_map.filename}")
            else:
                transform = transform[0]
            
            try:
                logger.info(f'Applying transform on {tsnr_map.filename}')
                # Resample tsnr map using ants
                tsnr_T1w = apply_transforms(
                    fixed=image_read(t1w.path),
                    moving=image_read(tsnr_map.path),
                    transformlist=[transform.path],
                    interpolation='linear'
                )
                # Save resampled map
                image_write(
                    tsnr_T1w, 
                    out_file
                )
            except:
                logger.info(f"Could not resample {tsnr_map.filename}")

def resample_to_MNI152(ds_name, tsnr_path, fmriprep_path, smriprep_path):
    logger = logging.getLogger(__name__)
    logger.info(f'generating tsnr maps from {Path(tsnr_path).name.split('.')[-1]} data for the {ds_name} dataset')
    logger.info(f"loading tSNR data: {tsnr_path}")
    layout_tsnr = bids.BIDSLayout(tsnr_path, validate=False, is_derivative=True)
    subjects = layout_tsnr.get_subjects()

    logger.info(f"loading fMRIPrep data: {fmriprep_path}")
    layout_fmriprep = bids.BIDSLayout(fmriprep_path, validate=False, is_derivative=True)

    logger.info(f"loading sMRIPrep data: {smriprep_path}")
    layout_smriprep = bids.BIDSLayout(smriprep_path, validate=False, is_derivative=True)

    for subject in subjects:
        # Retrieve T1w to MNI transformation
        to_mni = [f for f in layout_smriprep.get(subject=subject, suffix='xfm', extension='h5') if 'to-MNI152NLin2009cAsym' in f.filename]
        # Validate input
        if len(to_mni) == 0:
            raise IOError(f"No transformation from T1w to MNI file associated with {tsnr_map.filename}")
        elif len(to_mni) > 1:
            raise IOError(f"More than one transformation from T1w to MNI file associated with {tsnr_map.filename}")
        else:
            to_mni = to_mni[0]

        tsnr_maps = [f for f in layout_tsnr.get(subject=subject, suffix='statmap', extension='.nii.gz') if 'stat-tsnr' in f.filename and 'space' not in f.filename]

        for tsnr_map in tsnr_maps:
            entities = tsnr_map.get_entities()
            entities.update({
                'space': 'MNI152NLin2009cAsym',
                'stat': 'tsnr',
            })

            pattern = pattern="sub-{subject}/ses-{session}/{datatype}/sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}[_echo-{echo}]_stat-{stat}_desc-{desc}_{suffix}.nii.gz"
            out_file = layout_tsnr.build_path(entities, pattern, validate=False)

            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(out_file).exists():
                # Get reference images from fMRIPrep directory
                ref = layout_fmriprep.get(
                    subject=entities['subject'],
                    session=entities['session'],
                    run=entities['run'],
                    space='MNI152NLin2009cAsym',
                    suffix='boldref',
                    extension='nii.gz'
                )
                # Validate input
                if len(ref) == 0:
                    raise IOError(f"No ref boldref file associated with {tsnr_map.filename}")
                elif len(ref) > 1:
                    raise IOError(f"More than one ref boldref file associated with {tsnr_map.filename}")
                else:
                    ref = ref[0]
                
                # Get transform from fMRIPrep directory
                transform = layout_fmriprep.get(
                    subject=entities['subject'],
                    session=entities['session'],
                    run=entities['run'],
                    desc='coreg',
                    suffix='xfm',
                    extension='txt'
                )
                # Validate input
                if len(transform) == 0:
                    raise IOError(f"No transform file associated with {tsnr_map.filename}")
                elif len(transform) > 1:
                    raise IOError(f"More than one transform file associated with {tsnr_map.filename}")
                else:
                    transform = transform[0]

                logger.info(f'Applying transform on {tsnr_map.filename}')
                # Resample tsnr map using ants
                tsnr_MNI = apply_transforms(
                    fixed=image_read(ref.path),
                    moving=image_read(tsnr_map.path),
                    transformlist= [to_mni.path, transform.path],
                    interpolation='lanczosWindowedSinc'
                )
                # Save resampled map
                image_write(
                    tsnr_MNI, 
                    out_file
                )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()