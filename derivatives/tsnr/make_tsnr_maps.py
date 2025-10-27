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
def main(ds_name, ds_path, output_filepath):
    tsnr_maps(ds_name, ds_path, output_filepath)

def tsnr_maps(ds_name, ds_path, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info(f'generating tsnr maps from fmriprep data for the {ds_name} dataset')
    logger.info(f"loading BIDS: {ds_name}")

    layout = bids.BIDSLayout(ds_path, validate=False)
    bolds = layout.get(suffix='bold', extension='.nii.gz', desc='preproc')

    for bold in bolds:
        tsnr_path = bold.path.replace(ds_path,output_filepath).replace('_part-mag','').replace('desc-preproc','stat-tsnr').replace('bold','statmap')
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