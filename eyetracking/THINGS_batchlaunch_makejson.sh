#!/bin/bash -l

#module load python/3.7 #?? on elm?

#source /home/mariestl/my_envs/pupil_venv/bin/activate
source /data/neuromod/virtualenvs/eyetracking/bin/activate

export LD_LIBRARY_PATH=/data/neuromod/virtualenvs/eyetracking/lib

CODEDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"
DATADIR="/data/neuromod/DATA/fmri_tmp/things/sourcedata"
OUTDIR="/home/mariestl/cneuromod/THINGS/Eye-tracking/offline_calibration"

for SUBID in sub-01 sub-02 sub-03 sub-06
do
  for SESID in {2..20}
  do
    python -m THINGS_make_config_json \
        --code_dir="${CODEDIR}" \
        --data_dir="${DATADIR}" \
        --out_dir="${OUTDIR}" \
        --sub=${SUBID} \
        --ses=${SESID}
  done
done
