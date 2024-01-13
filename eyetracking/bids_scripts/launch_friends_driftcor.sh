
#source /data/neuromod/virtualenvs/eyetracking/bin/activate
source /home/mariestl/my_envs/things_venv/bin/activate
# run on elm...
DATADIR="/unf/eyetracker/neuromod/friends/sourcedata"
OUTDIR="/data/neuromod/projects/eyetracking_bids/friends"
MKVDIR="/data/neuromod/DATA/cneuromod/friends/stimuli"

python -m friends_driftCor \
      --in_path="${DATADIR}" \
      --mkv_path="${MKVDIR}" \
      --out_path="${OUTDIR}"
