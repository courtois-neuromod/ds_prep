#!/bin/bash -l

# Local (Marie laptop)
source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

CODEDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
DATADIR="/home/labopb/Documents/Marie/neuromod/triplets/eyetracking"

python -m TP_make_config_json_TP \
      --code_dir="${CODEDIR}" \
      --data_dir="${DATADIR}" \
      --sub="sub-03" \
      --ses="ses-001"
