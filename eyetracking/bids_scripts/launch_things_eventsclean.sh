
source /data/neuromod/virtualenvs/eyetracking/bin/activate
# run on elm...
ETDIR="/data/neuromod/projects/eyetracking_bids/things/Events_files_enhanced"
BEHAVDIR="/data/neuromod/temp_marie/things_events/clean_files"
RUNDIR="/home/mariestl/cneuromod/ds_prep/eyetracking"
OUTDIR="/data/neuromod/projects/eyetracking_bids/things/Events_files_final"

python -m things_add_etQC_2cleanEvents \
      --et_path="${ETDIR}" \
      --behav_path="${BEHAVDIR}" \
      --run_dir="${RUNDIR}" \
      --out_path="${OUTDIR}"
