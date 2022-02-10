#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

RUNDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"

for cfile in ${RUNDIR}/config/config_THINGS/config_THINGS_s*.json; do
  python -m quality_check_THINGS_summary \
        --run_dir="${RUNDIR}" \
        --config="${cfile}" \
done
