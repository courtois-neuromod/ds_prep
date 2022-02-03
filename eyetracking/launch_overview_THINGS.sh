#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

RUNDIR="/home/labopb/Documents/Marie/neuromod/THINGS/Eye-tracking/offline_calibration"

python -m quality_check_THINGS_overview \
      --idir="${RUNDIR}" \
      --odir="${RUNDIR}/overview"
