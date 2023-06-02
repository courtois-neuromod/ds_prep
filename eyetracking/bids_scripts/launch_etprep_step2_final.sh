
source /data/neuromod/virtualenvs/eyetracking/bin/activate
# run on elm...
TASK="${1}" # emotionsvideos, floc, friends, mario, mario3, mariostars, retino, things, triplets
DATADIR="/unf/eyetracker/neuromod/${TASK}/sourcedata"
RUNDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"
OUTDIR="/data/neuromod/projects/eyetracking_bids/${TASK}"

python -m et_prep_step2 \
      --in_path="${DATADIR}" \
      --run_dir="${RUNDIR}" \
      --is_final \
      --out_path="${OUTDIR}"
