#!/bin/bash -l

#module load python/3.7 #?? on elm?

#source /home/mariestl/my_envs/pupil_venv/bin/activate
source /data/neuromod/virtualenvs/eyetracking/bin/activate

export LD_LIBRARY_PATH=/data/neuromod/virtualenvs/eyetracking/lib

RUNDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"

for cfile in ${RUNDIR}/config/config_THINGS/config_THINGS_s*.json; do
  python -m quality_check_THINGS_summary \
        --run_dir="${RUNDIR}" \
        --config="${cfile}"
done
