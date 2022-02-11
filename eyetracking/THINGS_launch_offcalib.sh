#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

RUNDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
config_file=${1}

python -m THINGS_offline_calibration \
      --run_dir="${RUNDIR}" \
      --config="${RUNDIR}/config/config_THINGS/${config_file}"
