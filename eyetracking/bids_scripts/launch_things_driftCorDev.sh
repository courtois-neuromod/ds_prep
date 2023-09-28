
source /data/neuromod/virtualenvs/eyetracking/bin/activate
# run on elm...
PHASE="${1}" # 1, 2, 3
DATADIR="/unf/eyetracker/neuromod/things/sourcedata"
OUTDIR="/data/neuromod/projects/eyetracking_bids/things"

python -m things_driftCor_Dev \
  --in_path="${DATADIR}" \
  --phase_num="${PHASE}" \
  --out_path="${OUTDIR}"
