import os, glob
from pathlib import Path

import click
from nilearn.image import mean_img


@click.command()
@click.argument('ds_name', type=str)
@click.argument('ds_path', type=click.Path())
@click.option('--avg_MNI', is_flag=True)
@click.option('--avg_T1w', is_flag=True)
def main(ds_name, ds_path, avg_MNI, avg_T1w):

    for sub_path in glob.glob(f"{ds_path}/sub*"):
        sub_num = os.path.basename(sub_path)
        
        if avg_MNI:
            sub_list_MNI = sorted(glob.glob(f"{ds_path}/{sub_num}/ses*/func/*MNI152NLin2009cAsym*stat-tsnr*nii.gz"))
            sub_mean_MNI = mean_img(sub_list_MNI, copy_header=True)
            sub_mean_MNI.to_filename(
                f"{ds_path}/{sub_num}/{sub_num}_task-{ds_name}_space-MNI152NLin2009cAsym_stat-avgtsnr_statmap.nii.gz"
            )
        if avg_T1w:
            sub_list_t1w = sorted(glob.glob(f"{ds_path}/{sub_num}/ses*/func/*T1w*stat-tsnr*nii.gz"))
            sub_mean_t1w = mean_img(sub_list_t1w, copy_header=True)
            sub_mean_t1w.to_filename(
                f"{ds_path}/{sub_num}/{sub_num}_task-{ds_name}_space-T1w_stat-avgtsnr_statmap.nii.gz"
            )


if __name__ == '__main__':

    main()