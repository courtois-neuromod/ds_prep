#!/bin/bash -l

# Conda env to run script locally:
#conda activate movie_making

GAZEPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-01/ses-039/run_s2e04a_online_gaze2D.npz"
FILMPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/video_stimuli/friends_s2e04a_copy.mkv"
OUTPATH="/home/labopb/Documents/Marie/neuromod/eyetrack_movies/gaze_movies"

python -m FRIENDS_Gaze_on_Film \
      --gaze="${GAZEPATH}" \
      --film="${FILMPATH}" \
      --out_path="${OUTPATH}" \
      --fps=29.97 \
      --partial \
      --start_frame=15000 \
      --num_frames=1000 \
      --gaze_confthres=0.98 \
      --sub="s1_online" \
      --ses="s2e04a"
