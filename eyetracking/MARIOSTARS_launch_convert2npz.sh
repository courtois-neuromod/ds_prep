#!/bin/bash -l

#module load python/3.7 #?? on elm?

#source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

CODEDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"

SUB_NUM="${1}" # 01, 02, 03
SES_NUM="${2}" # 001, 002, 003

INDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/pupil_data/sub-${SUB_NUM}/ses-${SES_NUM}"
OUTDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/offline_calibration/sub-${SUB_NUM}/ses-${SES_NUM}"

python -m MARIOSTARS_convert_serialGazeAndFix_2_npz \
      --in_path="${INDIR}" \
      --code_dir="${CODEDIR}" \
      --out_path="${OUTDIR}"
