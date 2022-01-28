#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/mariestl/my_envs/pupil_venv/bin/activate

RUNDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"

python -m check_file_order \
      --run_dir="${RUNDIR}" \
      --sub="sub-01" \
      --ses="ses-004"
