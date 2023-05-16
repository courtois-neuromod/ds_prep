#!/bin/bash -l

# Conda env to run script locally:
#conda activate movie_making

GAZEPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/deepgaze_coord/fullsize/friends_s2e04a_coord.tsv"
FILMPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/video_stimuli/friends_s2e04a_copy.mkv"
OUTPATH="/home/labopb/Documents/Marie/neuromod/eyetrack_movies/gaze_movies"

# fps ~ 29.97
python -m FRIENDS_Gaze_on_Film \
      --gaze="${GAZEPATH}" \
      --film="${FILMPATH}" \
      --out_path="${OUTPATH}" \
      --deepgaze \
      --outname="s2e04a_DeepGaze"
