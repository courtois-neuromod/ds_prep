#!/bin/bash -l

#module load python/3.7 #?? on elm?

#source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

CODEDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
INDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/pupil_data/sub-01/ses-003"
OUTDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/offline_calibration/sub-01/ses-003"

python -m MARIOSTARS_convert_serialGazeAndFix_2_npz \
      --in_path="${INDIR}" \
      --code_dir="${CODEDIR}" \
      --out_path="${OUTDIR}" \
      --fixations
