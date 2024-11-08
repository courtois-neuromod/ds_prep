import os
import json
import bids
import argparse
from pathlib import Path
import logging
import nibabel as nb
import numpy as np
import scipy.ndimage
import datalad.api
from datalad.support.annexrepo import AnnexRepo
from deepbrain import Extractor
import scipy.ndimage.morphology

from dipy.align.imaffine import (
    transform_centers_of_mass,
    AffineMap,
    MutualInformationMetric,
    AffineRegistration,
    VerbosityLevels,
)
from dipy.align.transforms import (
    AffineTransform3D,
    RigidTransform3D,
    RigidScalingTransform3D,
    RigidIsoScalingTransform3D,
)

PYBIDS_CACHE_PATH = ".pybids_cache"
MNI_PATH = "../../global/templates/MNI152_T1_1mm.nii.gz"
MNI_MASK_PATH = "../../global/templates/MNI152_T1_1mm_brain.nii.gz"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="deface anatomical series by performing an affine registration to a template and warping mask to native space",
    )

    parser.add_argument("bids_path", help="BIDS folder to deface.")
    parser.add_argument(
        "--participant-label",
        action="store",
        nargs="+",
        help="a space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed)",
    )
    parser.add_argument(
        "--session-label",
        action="store",
        nargs="+",
        help="a space delimited list of sessions identifiers or a single "
        "identifier (the ses- prefix can be removed)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force pyBIDS reset_database and reindexing",
    )
    parser.add_argument(
        "--datalad",
        action="store_true",
        help="Update distribution-restrictions metadata and commit changes",
    )
    parser.add_argument(
        "--save-all-masks",
        action="store_true",
        help="Save mask for all defaced series, default is only saving mask for reference serie.",
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Output debug images in the current directory",
    )
    parser.add_argument(
        "--ref-bids-filters",
        dest="ref_bids_filters",
        action="store",
        type=_bids_filter,
        help="path to or inline json with pybids filters to select session reference to register defacemask",
    )
    parser.add_argument(
        "--other-bids-filters",
        dest="other_bids_filters",
        action="store",
        type=_bids_filter,
        help="path to or inline json with pybids filters to select all images to deface",
    )
    parser.add_argument(
        "--debug",
        dest="debug_level",
        action="store",
        default="info",
        help="debug level",
    )
    return parser.parse_args()


def _filter_pybids_any(dct):
    return {k: bids.layout.Query.ANY if v == "*" else v for k, v in dct.items()}


def _bids_filter(json_str):
    if os.path.exists(os.path.abspath(json_str)):
        json_str = Path(json_str).read_text()
    return json.loads(json_str, object_hook=_filter_pybids_any)


def registration(ref, moving, ref_mask=None, moving_mask=None):
    ref_mask_data, mov_mask_data = None, None
    ref_data = ref.get_fdata()
    if ref_mask:
        ref_mask_data = (ref_mask.get_fdata() > 0.5).astype(np.int32)
    mov_data = moving.get_fdata()
    if moving_mask:
        mov_mask_data = (moving_mask.get_fdata() > 0.5).astype(np.int32)

    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    transform = RigidTransform3D()
    affreg = AffineRegistration(
        metric=metric, level_iters=[10000, 1000, 0], factors=[6, 4, 2], sigmas=[4, 2, 0]
    )
    rigid = affreg.optimize(
        ref_data,
        mov_data,
        transform,
        None,
        ref.affine,
        moving.affine,
        starting_affine="mass",
        static_mask=ref_mask_data,
        moving_mask=mov_mask_data,
    )

    affreg = AffineRegistration(
        metric=metric, level_iters=[10000, 1000, 0], factors=[4, 2, 2], sigmas=[4, 2, 0]
    )
    transform = RigidScalingTransform3D()
    # transform = AffineTransform3D()
    return affreg.optimize(
        ref_data,
        mov_data,
        transform,
        None,
        ref.affine,
        moving.affine,
        starting_affine=rigid.affine,
        static_mask=ref_mask_data,
        moving_mask=mov_mask_data,
    )


def output_debug_images(ref, moving, affine):
    moving_nb = moving.get_image()
    moving_suffix = moving.entities["suffix"]
    moving_reg_path = moving.path.replace(
        f"_{moving_suffix}", f"_space-MNIlinreg_{moving_suffix}"
    )
    moving_reg = affine.transform(
        moving_nb.get_fdata(),
        image_grid2world=moving_nb.affine,
        sampling_grid_shape=ref.shape,
        sampling_grid2world=ref.affine,
    )
    logging.info(
        f"writing reference serie linearly warped to MNI template: {moving_reg_path}"
    )
    nb.Nifti1Image(moving_reg, ref.affine).to_filename(moving_reg_path)

    ref_inv_path = moving.path.replace(
        f"_{moving_suffix}", f"_mod-{moving_suffix}_MNIlinreg"
    )
    ref_inv = affine.transform_inverse(
        ref.get_fdata(),
        image_grid2world=ref.affine,
        sampling_grid_shape=moving_nb.shape,
        sampling_grid2world=moving_nb.affine,
    )
    logging.info(
        f"writing MNI template image linearly warped to the reference serie: {ref_inv_path}"
    )
    nb.Nifti1Image(ref_inv, moving_nb.affine).to_filename(ref_inv_path)


def warp_mask(tpl_mask, target, affine):
    matrix = np.linalg.inv(tpl_mask.affine).dot(affine.affine_inv.dot(target.affine))
    warped_mask = scipy.ndimage.affine_transform(
        np.asanyarray(tpl_mask.dataobj).astype(np.int32),
        matrix,
        output_shape=target.shape,
        mode="nearest",
    )
    return nb.Nifti1Image(warped_mask, target.affine)


def main():

    args = parse_args()
    logging.basicConfig(level=logging.getLevelName(args.debug_level.upper()))

    pybids_cache_path = os.path.join(args.bids_path, PYBIDS_CACHE_PATH)

    layout = bids.BIDSLayout(
        args.bids_path,
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        index_metadata=False,
        validate=False,
    )

    if args.datalad:
        annex_repo = AnnexRepo(args.bids_path)

    subject_list = (
        args.participant_label if args.participant_label else bids.layout.Query.ANY
    )
    session_list = args.session_label if args.session_label else bids.layout.Query.ANY
    filters = dict(
        subject=subject_list,
        session=session_list,
        **args.ref_bids_filters,
        extension=['nii','nii.gz'])
    deface_ref_images = layout.get(**filters)

    if not len(deface_ref_images):
        logging.info(f"no reference image found with condition {filters}")
        return

    new_files, modified_files = [], []

    script_dir = os.path.dirname(__file__)

    mni_path = os.path.abspath(os.path.join(script_dir, MNI_PATH))
    mni_mask_path = os.path.abspath(os.path.join(script_dir, MNI_MASK_PATH))
    # if the MNI template image is not available locally
    if not os.path.exists(os.path.realpath(mni_path)):
        datalad.api.get(mni_path, dataset=datalad.api.Dataset(script_dir + "/../../"))
    tmpl_image = nb.load(mni_path)
    tmpl_image_mask = nb.load(mni_mask_path)
    tmpl_defacemask = generate_deface_ear_mask(tmpl_image)
    brain_xtractor = Extractor()

    for ref_image in deface_ref_images:
        subject = ref_image.entities["subject"]
        session = ref_image.entities["session"]

        datalad.api.get(ref_image.path)
        ref_image_nb = ref_image.get_image()

        matrix_path = ref_image.path.replace(
            "_%s%s" % (ref_image.entities["suffix"], ref_image.entities["extension"]),
            "_mod-%s_defacemaskreg.mat" % ref_image.entities["suffix"],
        )

        if os.path.exists(matrix_path):
            logging.info("reusing existing registration matrix")
            ref2tpl_affine = AffineMap(np.loadtxt(matrix_path))
        else:
            logging.info(f"running registration of reference serie: {ref_image.path}")
            brain_mask = (brain_xtractor.run(ref_image_nb.get_fdata()) > 0.99).astype(
                np.uint8
            )
            brain_mask[:] = scipy.ndimage.binary_dilation(
                brain_mask, iterations=4
            )
            brain_mask_nb = nb.Nifti1Image(brain_mask, ref_image_nb.affine)
            ref2tpl_affine = registration(
                tmpl_image, ref_image_nb, tmpl_image_mask, brain_mask_nb
            )
            np.savetxt(matrix_path, ref2tpl_affine.affine)
            new_files.append(matrix_path)

        if args.debug_images:
            output_debug_images(tmpl_image, ref_image, ref2tpl_affine)

        series_to_deface = []
        for filters in args.other_bids_filters:
            series_to_deface.extend(
                layout.get(
                    extension=["nii", "nii.gz"],
                    subject=subject,
                    session=session,
                    **filters,
                )
            )


        for serie in series_to_deface:
            if args.datalad:
                if (
                    next(annex_repo.get_metadata(serie.path))[1].get(
                        "distribution-restrictions"
                    )
                    is None
                ):
                    logging.info(
                        f"skip {serie.path} as there are no distribution restrictions metadata set."
                    )
                    continue
            logging.info(f"defacing {serie.path}")

            datalad.api.get(serie.path)
            # unlock before making any change to avoid unwanted save
            if args.datalad:
                annex_repo.unlock([serie.path for serie in series_to_deface])

            serie_nb = serie.get_image()
            warped_mask = warp_mask(tmpl_defacemask, serie_nb, ref2tpl_affine)
            if args.save_all_masks or serie == ref_image:
                warped_mask_path = serie.path.replace(
                    "_%s" % serie.entities["suffix"],
                    "_mod-%s_defacemask" % serie.entities["suffix"],
                )
                if os.path.exists(warped_mask_path):
                    logging.warning(
                        f"{warped_mask_path} already exists : will not overwrite, clean before rerun"
                    )
                else:
                    warped_mask.to_filename(warped_mask_path)
                    new_files.append(warped_mask_path)

            masked_serie = nb.Nifti1Image(
                np.asanyarray(serie_nb.dataobj) * np.asanyarray(warped_mask.dataobj),
                serie_nb.affine,
                serie_nb.header,
            )
            masked_serie.to_filename(serie.path)
            modified_files.append(serie.path)

    if args.datalad and len(modified_files):
        logging.info("saving files and metadata changes in datalad")
        annex_repo.set_metadata(
            modified_files, remove={"distribution-restrictions": "sensitive"}
        )
        datalad.api.save(
            modified_files + new_files,
            message="__deface__ %d series/images and update distribution-restrictions"
            % len(modified_files),
        )


# generates the mask on the fly from the template image, using hard-coded markers
# the mask image is larger that the template to include the full face and allow processing
# of images with larger FoV (eg. cspine acquisitions)
def generate_deface_ear_mask(mni):

    deface_ear_mask = np.ones(np.asarray(mni.shape) * (1, 1, 2), dtype=np.uint8)
    deface_ear_mask[:, :, :mni.shape[2]] = 0
    affine_ext = mni.affine.copy()
    affine_ext[2, -1] -= mni.shape[-1]

    above_eye_marker = [218, 240]
    jaw_marker = [130, 182]
    ear_marker = [25, 160]
    ear_marker2 = [5, 260]

    # remove face
    deface_ear_mask[:, jaw_marker[0] :, : jaw_marker[1]] = 0
    y_coords = np.round(
        np.linspace(
            jaw_marker[0], above_eye_marker[0], above_eye_marker[1] - jaw_marker[1]
        )
    ).astype(np.int32)
    for z, y in zip(range(jaw_marker[1], above_eye_marker[1]), y_coords):
        deface_ear_mask[:, y:, z] = 0

    # remove ears
    deface_ear_mask[: ear_marker[0], :, : ear_marker[1]] = 0
    deface_ear_mask[-ear_marker[0] :, :, : ear_marker[1]] = 0
    x_coords = np.round(
        np.linspace(ear_marker[0], ear_marker2[0], ear_marker2[1] - ear_marker[1])
    ).astype(np.int32)
    for z, x in zip(range(ear_marker[1], ear_marker2[1]), x_coords):
        deface_ear_mask[:x, :, z] = 0
        deface_ear_mask[-x:, :, z] = 0

    # remove data on the image size where the body doesn't extend
    deface_ear_mask[-1] = 0
    deface_ear_mask[0] = 0
    deface_ear_mask[:, -1, :] = 0
    deface_ear_mask[:, :, -1] = 0

    return nb.Nifti1Image(deface_ear_mask, affine_ext)


if __name__ == "__main__":
    main()
