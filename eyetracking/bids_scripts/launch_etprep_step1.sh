
source /data/neuromod/virtualenvs/eyetracking/bin/activate
# run on elm...
DATADIR="/unf/eyetracker/neuromod/triplets/sourcedata"
RUNDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"
OUTDIR="/home/mariestl/cneuromod/eyetracking/triplets"

python -m et_prep_step1 \
      --in_path="${DATADIR}" \
      --run_dir="${RUNDIR}" \
      --out_path="${OUTDIR}"
