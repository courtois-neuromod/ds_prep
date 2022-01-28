#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/mariestl/my_envs/pupil_venv/bin/activate

RUNDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"
config_file=${1}

python -m quality_check_THINGS \
      --run_dir="${RUNDIR}" \
      --config="${RUNDIR}/config/config_THINGS/${config_file}"
