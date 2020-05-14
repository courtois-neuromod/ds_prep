import os
import json
import bids
import argparse
from pathlib import Path
import subprocess
import logging
import nibabel as nb
import numpy as np
import scipy.ndimage
import datalad.api
from datalad.support.annexrepo import AnnexRepo

from dipy.align.imaffine import (transform_centers_of_mass,
                                  AffineMap,
                                  MutualInformationMetric,
                                  AffineRegistration)
from dipy.align.transforms import (AffineTransform3D, RigidTransform3D)

from nipype.interfaces import fsl


PYBIDS_CACHE_PATH = '.pybids_cache'
MNI_PATH = '../../global/templates/MNI152_T1_1mm.nii.gz'
DEFACE_MASK_PATH = '../../global/templates/deface_ear_mask.nii.gz'


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    parser.add_argument('bids_path',
                   help='BIDS folder to convert.')
    parser.add_argument(
        '--participant-label', action='store', nargs='+',
        help='a space delimited list of participant identifiers or a single '
             'identifier (the sub- prefix can be removed)')
    parser.add_argument(
        '--force-reindex', action='store_true',
        help='Force pyBIDS reset_database and reindexing')
    parser.add_argument(
        '--debug-images', action='store_true',
        help='Output debug images in the current directory')
    parser.add_argument(
        '--ref-bids-filters', dest='ref_bids_filters', action='store',
        type=_bids_filter,
        help="path to or inline json with pybids filters to select session reference to register defacemask")
    parser.add_argument(
        '--other-bids-filters', dest='other_bids_filters', action='store',
        type=_bids_filter,
        help="path to or inline json with pybids filters to select all images to deface")
    return parser.parse_args()

def _filter_pybids_any(dct):
    return {k: bids.layout.Query.ANY if v == "*" else v for k, v in dct.items()}

def _bids_filter(json_str):
    if os.path.exists(os.path.abspath(json_str)):
        json_str = Path(json_str).read_text()
    return json.loads(json_str, object_hook=_filter_pybids_any)

def registration(ref, moving):
    ref_data = ref.get_fdata()
    mov_data = moving.get_fdata()
    c_of_mass = transform_centers_of_mass(ref_data, ref.affine,
                                          mov_data, moving.affine)
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    sigmas = [5.0, 3.0, 1.0]
    factors = [8, 4, 2]
    level_iters = [10000, 1000, 100]
    transform = RigidTransform3D()
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    rigid = affreg.optimize(ref_data, mov_data, transform, None,
                            ref.affine, moving.affine,
                            starting_affine=c_of_mass.affine)
    transform = AffineTransform3D()
    return affreg.optimize(ref_data, mov_data, transform, None,
                           ref.affine, moving.affine,
                           starting_affine=rigid.affine)


def output_debug_images(ref, moving, affine):
    moving_reg = affine.transform(
        moving.get_fdata(),
        image_grid2world=moving.affine,
        sampling_grid_shape=ref.shape,
        sampling_grid2world=ref.affine)
    nb.Nifti1Image(moving_reg, ref.affine).to_filename('moving_reg.nii.gz')
    ref_inv = affine.transform_inverse(
        ref.get_fdata(),
        image_grid2world=ref.affine,
        sampling_grid_shape=moving.shape,
        sampling_grid2world=moving.affine)
    nb.Nifti1Image(ref_inv, moving.affine).to_filename('ref_inv.nii.gz')


def warp_mask(tpl_mask, target, affine):
    matrix = np.linalg.inv(tpl_mask.affine).dot(affine.affine_inv.dot(target.affine))
    warped_mask = scipy.ndimage.affine_transform(
        np.asanyarray(tpl_mask.dataobj).astype(np.int32),
        matrix,
        output_shape=target.shape,
        mode='nearest')
    return nb.Nifti1Image(warped_mask, target.affine)

def main():

    args = parse_args()

    pybids_cache_path = os.path.join(args.bids_path, PYBIDS_CACHE_PATH)

    layout = bids.BIDSLayout(
        args.bids_path,
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        index_metadata=False,
        validate=False)

    annex_repo = AnnexRepo(args.bids_path)

    subject_list = args.participant_label if args.participant_label else bids.layout.Query.ANY
    deface_ref_images = layout.get(
        subject=subject_list,
        **args.ref_bids_filters,
        extension='nii.gz')

    modified_files = []

    script_dir = os.path.dirname(__file__)
    tmpl_image = nb.load(os.path.join(script_dir, MNI_PATH))
    tmpl_defacemask = nb.load(os.path.join(script_dir, DEFACE_MASK_PATH))

    for ref_image in deface_ref_images:
        subject = ref_image.entities['subject']
        session = ref_image.entities['session']

        ref_image_nb = nb.load(ref_image)
        ref2tpl_affine = registration(tmpl_image, ref_image_nb)
        output_debug_images(tmpl_image, ref_image_nb, ref2tpl_affine)

        series_to_deface = []
        for filters in args.other_bids_filters:
            series_to_deface.extend(layout.get(
                extension='nii.gz',
                subject=subject, session=session, **filters))

        for serie in series_to_deface:
            #if next(annex_repo.get_metadata(serie.path))[1].get('distribution-restrictions') is None:
            #    continue

            datalad.api.unlock(serie.path)
            warped_mask_path = serie.path.replace(
                '_%s'%serie.entities['suffix'],
                '_mod-%s_defacemask'%serie.entities['suffix'])

            serie_nb = serie.get_image()
            warped_mask = warp_mask(tmpl_defacemask, serie_nb, ref2tpl_affine)
            warped_mask.to_filename(warped_mask_path)

            masked_serie = nb.Nifti1Image(
                np.asanyarray(serie_nb.dataobj) * np.asanyarray(warped_mask.dataobj),
                serie_nb.affine,
                serie_nb.header)
            masked_serie.to_filename(serie.path)
            modified_files.append(serie.path)

    #datalad.api.add(modified_files)
    if len(modified_files):
        #annex_repo.set_metadata(modified_files, remove={'distribution-restrictions': 'sensitive'})


if __name__ == "__main__":
    main()

def generate_deface_ear_mask():

    mni = nb.load(MNI_PATH)
    deface_ear_mask = np.ones(np.asarray(mni.shape)*(1,1,2),dtype=np.int8)
    affine_ext=mni.affine.copy()
    affine_ext[2,-1]-=mni.shape[-1]

    above_eye_marker = [218,245]
    jaw_marker = [126,182]
    ear_marker = [30,170]
    ear_marker2 = [5,300]

    # remove face
    deface_ear_mask[:,jaw_marker[0]:,:jaw_marker[1]] = 0
    y_coords = np.round(np.linspace(jaw_marker[0],above_eye_marker[0],above_eye_marker[1]-jaw_marker[1])).astype(np.int)
    for z,y in zip(range(jaw_marker[1],above_eye_marker[1]),y_coords):
        deface_ear_mask[:,y:,z]=0

    # remove ears
    deface_ear_mask[:ear_marker[0],:,:ear_marker[1]]=0
    deface_ear_mask[-ear_marker[0]:,:,:ear_marker[1]]=0
    x_coords=np.round(np.linspace(ear_marker[0],ear_marker2[0],ear_marker2[1]-ear_marker[1])).astype(np.int)
    for z,x in zip(range(ear_marker[1],ear_marker2[1]),x_coords):
        deface_ear_mask[:x,:,z]=0
        deface_ear_mask[-x:,:,z]=0

    # remove data on the image size where the body doesn't extend
    deface_ear_mask[-1]=0
    deface_ear_mask[0]=0
    deface_ear_mask[:,-1,:]=0
    deface_ear_mask[:,:,-1]=0

    nb.Nifti1Image(deface_ear_mask, affine_ext).to_filename('deface_ear_mask.nii.gz')
