#!/bin/bash -l

#module load python/3.7 #?? on elm?

source /home/labopb/Documents/Marie/neuromod/pupil_venv/bin/activate

export LD_LIBRARY_PATH=/data/neuromod/virtualenvs/eyetracking/lib

CODEDIR="/home/labopb/Documents/Marie/neuromod/ds_prep/eyetracking"
DATADIR="/home/labopb/Documents/Marie/neuromod/THINGS/Eye-tracking"

for SUBID in sub-01 sub-02 sub-03 sub-06
do
  for SESID in {3..20}
  do
    python -m THINGS_make_config_json \
        --code_dir="${CODEDIR}" \
        --data_dir="${DATADIR}" \
        --sub=${SUBID} \
        --ses=${SESID}
  done
done
