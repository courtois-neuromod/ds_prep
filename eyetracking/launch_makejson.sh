#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

CODEDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"
DATADIR="/data/neuromod/DATA/fmri_tmp/things/sourcedata"
OUTDIR="/home/mariestl/cneuromod/THINGS/Eye-tracking/offline_calibration"

python -m make_config_json \
      --code_dir="${CODEDIR}" \
      --data_dir="${DATADIR}" \
      --out_dir="${OUTDIR}" \
      --sub="sub-01" \
      --ses="ses-005"
