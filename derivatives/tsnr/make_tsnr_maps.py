# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path

import bids
import click
from nipype.algorithms.confounds import TSNR


@click.command()
@click.argument('ds_name', type=str)
@click.argument('ds_path', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.option('--echo', type=str, default=None, help='Specify the echo value to compute tSNR maps from fMRIPrep outputs')
@click.option('--me', is_flag=True, help='Flag to specify if working with tedana outputs. If specified, will compute tSNR maps for optcom and denoised images')
def main(ds_name, ds_path, output_filepath, echo, me):
    tsnr_maps(ds_name, ds_path, output_filepath, echo, me)

def tsnr_maps(ds_name, ds_path, output_filepath, echo=None, me=False):
    logger = logging.getLogger(__name__)
    logger.info(f'generating tsnr maps from fmriprep data for the {ds_name} dataset')
    logger.info(f"loading BIDS: {ds_name}")

    layout = bids.BIDSLayout(ds_path, validate=False)
    if me:
        # Get optimally combined and denoised images
        bolds_optcom = layout.get(suffix='bold', extension='.nii.gz', desc='optcom')
        bolds_denoised = layout.get(suffix='bold', extension='.nii.gz', desc='denoised')
        bolds = [*bolds_optcom, *bolds_denoised]
    else:
        bolds = layout.get(suffix='bold', extension='.nii.gz', desc='preproc', echo=echo)

    for bold in bolds:
        tsnr_path = bold.path.replace(ds_path,output_filepath).replace('_part-mag','').replace(f'desc-{bold.get_entities()["desc"]}',f'stat-tsnr_desc-{bold.get_entities()["desc"]}').replace('bold','statmap')

        Path(tsnr_path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(tsnr_path).exists():
            try:
                tsnr_if = TSNR(
                    in_file=bold.path,
                    tsnr_file=tsnr_path,
                    stddev_file=tsnr_path.replace('stat-tsnr', 'stat-stdev'),
                    mean_file=tsnr_path.replace('stat-tsnr', 'stat-mean'),
                )
                tsnr_if.run()
                del tsnr_if
            except:
                logger.info(f"could not process {os.path.basename(bold.path)}")

    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()