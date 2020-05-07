import os
import bids
import argparse
import tempfile
import subprocess
import logging
import datalad.api
from datalad.support.annexrepo import AnnexRepo
from pkg_resources import resource_filename, Requirement

from nipype.interfaces import fsl

PYBIDS_CACHE_PATH = '.pybids_cache'
PYDEFACE_SINGULARITY = 'pydeface-2.0.0.simg'

deface_ref_image = {
    'datatype':'anat',
    'suffix':'T1w',
    'rec':'norm',
    'acq':'',
    'bp':''}

other_images_to_deface = [
 {'datatype':'anat', 'bp':None, 'suffix': ['T2w','MP2RAGE','UNIT1']},
 {'datatype':'anat', 'bp':'cspine', 'suffix':['T2w']},
]

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    parser.add_argument('bids_path',
                   help='BIDS folder to convert.')
    parser.add_argument('--force-reindex', action='store_true',
                   help='Force pyBIDS reset_database and reindexing')
    return parser.parse_args()

def main():
    args = parse_args()

    pybids_cache_path = os.path.join(args.bids_path, PYBIDS_CACHE_PATH)

    layout = bids.BIDSLayout(
        args.bids_path,
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        index_metadata=False)
    repo = AnnexRepo(args.bids_path)

    deface_ref_images = layout.get(**deface_ref_image, extension='nii.gz')

    modified_files = []

    tmpl_defacemask = resource_filename(Requirement.parse("pydeface"),
                                        "pydeface/data/facemask.nii.gz")

    for ref_image in deface_ref_images:
        # defaced file already exists
        defacemask_path = ref_image.path.replace('_T1w','_mod-T1w_defacemask')
        template_reg_mat = ref_image.path.replace('.nii.gz','_pydeface.mat')
        if os.path.exists(defacemask_path):
            continue
        subject = ref_image.session
        session = ref_image.session

        cmd = ["pydeface", "--force", "--nocleanup", "--outfile", ref_image.path, ref_image.path ]
        datalad.api.unlock(ref_image.path)
        ret = subprocess.run(cmd)


        other_images = []
        for other_ents in other_images_to_deface:
            other_images.extend(layout.get(subject=subject, session=session, **other_ents))

        if ret.returncode==0:
            modified_files.append(anat.path)
        else:
            raise RuntimeError

        for other_image in other_images:
            datalad.api.unlock(other_image.path)
            warped_mask = ref_image.path.replace('_T1w','_mod-T1w_defacemask')

            flirt = fsl.FLIRT()
            flirt.inputs.in_file = tmpl_defacemask
            flirt.inputs.in_matrix_file = template_reg_mat
            flirt.inputs.apply_xfm = True
            flirt.inputs.interp = 'nearestneighbour'
            flirt.inputs.reference = other_image.path
            flirt.inputs.out_file = warped_mask
            flirt.inputs.output_type = '.nii.gz'
            flirt.run()

            # multiply mask by infile and save
            infile_img = load(other_image)
            warped_mask_img = load(warped_mask)
            outdata = other_image.get_data() * warped_mask_img.get_data()

            masked_brain = Nifti1Image(outdata, infile_img.get_affine(),
                                       infile_img.get_header())
            masked_brain.to_filename(other_image.path)
            modified_files.append(other_image.path)

    datalad.api.add(modified_files)
    repo.set_metadata(modified_files, remove={'distribution-restrictions': 'sensitive'})


if __name__ == "__main__":
    main()
