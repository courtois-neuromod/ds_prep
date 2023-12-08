
source /data/neuromod/virtualenvs/eyetracking/bin/activate
# run on elm...
ETDIR="/data/neuromod/projects/eyetracking_bids/things/final_bids_DriftCor"
EVDIR="/data/neuromod/projects/eyetracking_bids/things/Events_files_final"
OUTDIR="/data/neuromod/projects/eyetracking_bids/things/analyses_fixation"
SUBJECT_NUM="${1}" # 01, 02, 03

python -m things_analyzeFix \
  --sub_num="sub-${SUBJECT_NUM}"
  --et_path="${ETDIR}" \
  --ev_path="${EVDIR}" \
  --out_path="${OUTDIR}"
