#!/bin/bash -l

#module load python/3.7 #?? on elm?

#source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

EYETRACKDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/pupil_data/sub-01/ses-002"
GAZEDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/Eye-tracking/offline_calibration/sub-01/ses-002"
OUTDIR="/home/labopb/Documents/Marie/neuromod/MarioStars/test"

python -m MARIOSTARS_makemovie \
      --file_path="${EYETRACKDIR}" \
      --gaze_path="${GAZEDIR}" \
      --out_path="${OUTDIR}" \
      --driftcorr \
      --conf=0.80 \
      --fixconf=0.90
