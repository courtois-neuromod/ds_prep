#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /data/neuromod/virtualenvs/eyetracking/bin/activate

export LD_LIBRARY_PATH=/data/neuromod/virtualenvs/eyetracking/lib

RUNDIR="/home/mariestl/cneuromod/THINGS/Eye-tracking/offline_calibration"

python -m quality_check_THINGS_overview \
      --idir="${RUNDIR}" \
      --odir="${RUNDIR}/overview"
