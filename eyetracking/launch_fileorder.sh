#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

RUNDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
config_file=${1}

python -m check_file_order \
      --run_dir="${RUNDIR}" \
      --sub="sub-01" \
      --ses="ses-004"
