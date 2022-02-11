#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

RUNDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
config_file=${1}
# e.g., config_qc_s01_ses39_cal_1.json

python -m FRIENDS_quality_check \
      --run_dir="${RUNDIR}" \
      --config="${RUNDIR}/config/config_friends/qc/${config_file}"
