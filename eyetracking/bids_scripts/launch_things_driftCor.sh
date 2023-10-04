
source /data/neuromod/virtualenvs/eyetracking/bin/activate
# run on elm...
DATADIR="/unf/eyetracker/neuromod/things/sourcedata"
OUTDIR="/data/neuromod/projects/eyetracking_bids/things"

python -m things_driftCor \
  --in_path="${DATADIR}" \
  --out_path="${OUTDIR}"
