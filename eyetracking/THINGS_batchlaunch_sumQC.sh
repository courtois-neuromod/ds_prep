#!/bin/bash -l

#module load python/3.7 #?? on elm?

#source /home/mariestl/my_envs/pupil_venv/bin/activate
source /data/neuromod/virtualenvs/eyetracking/bin/activate

export LD_LIBRARY_PATH=/data/neuromod/virtualenvs/eyetracking/lib

CODEDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"

for cfile in ${CODEDIR}/config/config_THINGS/config_THINGS_s*.json; do
  python -m THINGS_qualitycheck_summary \
        --code_dir="${CODEDIR}" \
        --config="${cfile}"
done
