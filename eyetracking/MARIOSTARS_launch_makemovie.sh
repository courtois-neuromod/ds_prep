#!/bin/bash -l

#module load python/3.7 #?? on elm?

#source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate
SUB_NUM="${1}" # 01, 02, 03
SES_NUM="${2}" # 001, 002, 003

EYETRACKDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/pupil_data/sub-${SUB_NUM}/ses-${SES_NUM}"
GAZEDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/offline_calibration/sub-${SUB_NUM}/ses-${SES_NUM}"
OUTDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/test"

python -m MARIOSTARS_makemovie \
      --file_path="${EYETRACKDIR}" \
      --gaze_path="${GAZEDIR}" \
      --out_path="${OUTDIR}" \
      --driftcorr \
      --conf=0.80 \
      --fixconf=0.90
