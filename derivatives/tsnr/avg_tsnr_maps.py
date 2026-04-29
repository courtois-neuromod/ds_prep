import logging
from pathlib import Path

import bids
import click
from nilearn.image import mean_img


@click.command()
@click.argument('ds_name', type=str)
@click.argument('ds_path', type=click.Path())
@click.option('--avg_mni', is_flag=True)
@click.option('--avg_t1w', is_flag=True)
@click.option('--echo', type=str, default=None, help='Specify the echo value on which the tSNR maps were computed')
@click.option('--tedana', is_flag=True, help='Flag to specify if working with tedana outputs. If specified, will compute tSNR maps for optcom and denoised images')
def main(ds_name, ds_path, avg_mni, avg_t1w, echo, tedana):
    logger = logging.getLogger(__name__)
    logger.info(f'averaging tsnr maps from {Path(ds_path).name} data for the {ds_name} dataset')
    logger.info(f"loading tSNR maps: {ds_name}")

    ds_path = str(Path(ds_path))
    layout = bids.BIDSLayout(ds_path, is_derivative=True, validate=False)

    for sub in layout.get_subject():
        
        list_entities = []
        spaces = []

        entities = {
            'task': ds_name,
            'subject': sub,
            'suffix': 'statmap',
            'extension': '.nii.gz'
        }
        pattern = "sub-{subject}/sub-{subject}_task-{task}_space-{space}[_echo-{echo}]_stat-avgtsnr[_desc-{desc}]_statmap.nii.gz"

        if echo is not None:
            tmp_entities = entities.copy()
            tmp_entities.update({'echo': echo, 'desc': 'preproc'})
            list_entities.append(tmp_entities)
        if tedana:
            # Add optcom descriptor
            tmp_entities = entities.copy()
            tmp_entities.update({'desc': 'optcom'})
            list_entities.append(tmp_entities)
            # Add denoised descriptor
            tmp_entities = entities.copy()
            tmp_entities.update({'desc': 'denoised'})
            list_entities.append(tmp_entities)
        if not tedana and echo is None:
            list_entities.append(entities)
        
        if avg_mni:
            spaces.append('MNI152NLin2009cAsym')
        if avg_t1w:
            spaces.append('T1w')

        for elem_entities in list_entities:
            for space in spaces:
                elem_entities.update({'space': space})
                # Create output filename
                tmp_entities = elem_entities.copy()
                tmp_entities.update({'stat': 'avgtsnr'})
                out_file = layout.build_path(tmp_entities, pattern, validate=False)

                Path(out_file).parent.mkdir(parents=True, exist_ok=True)
                if not Path(out_file).exists():
                    # Retrieve files based on entities specified in `elem_entities`
                    sub_list = [f.get_image() for f in layout.get(**elem_entities) if 'stat-tsnr' in f.filename]
                    # Compute mean only if `sub_list` not empty
                    if len(sub_list)>0:
                        logger.info(f'Computing averaged tSNR map for {elem_entities["subject"]}')
                        sub_mean = mean_img(sub_list)
                        # Save averaged map
                        sub_mean.to_filename(out_file)
                        logger.info(f'Average computed for files with entities: \n{elem_entities}')
                        logger.info(f'Average maps saved : \n{out_file}')
                    else:
                        logger.info(f'Cannot compute average tsnr maps for files with entities: \n{elem_entities}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()