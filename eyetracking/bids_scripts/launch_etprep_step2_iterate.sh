
source /data/neuromod/virtualenvs/eyetracking/bin/activate
# run on elm...
TASK="${1}" # emotionsvideos, floc, friends, mario, mario3, mariostars, retino, things, triplets
DATADIR="/unf/eyetracker/neuromod/${TASK}/sourcedata"
RUNDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"
OUTDIR="/data/neuromod/projects/eyetracking_bids/${TASK}"
MKVDIR="/data/neuromod/DATA/cneuromod/friends/stimuli"

python -m et_prep_step2 \
      --in_path="${DATADIR}" \
      --task="${TASK}" \
      --out_path="${OUTDIR}"
