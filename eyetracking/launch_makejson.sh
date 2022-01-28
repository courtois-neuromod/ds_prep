#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

CODEDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
DATADIR="/home/labopb/Documents/Marie/neuromod/THINGS/Eye-tracking"

python -m make_config_json \
      --code_dir="${CODEDIR}" \
      --data_dir="${DATADIR}" \
      --sub="sub-01" \
      --ses="ses-005"
