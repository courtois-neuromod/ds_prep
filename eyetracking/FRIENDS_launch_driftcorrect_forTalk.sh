#!/bin/bash -l

# Conda env to run script locally:
#conda activate movie_making

RUNPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-02/ses-048"
EPISODE="06b"
GAZEPATH="${RUNPATH}/run_s6e${EPISODE}_online_gaze2D.npz"
FILMPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/video_stimuli/friends_s06e${EPISODE}.mkv"
DGPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/deepgaze_coord/s06e${EPISODE}_locmax_highestthres.npz"
OUTPATH="/home/labopb/Documents/Marie/hebart_lab/VSS_2022"

# fps ~ 29.97

python -m FRIENDS_driftcorrect_forTalk \
      --gaze="${GAZEPATH}" \
      --film="${FILMPATH}" \
      --deepgaze_file="${DGPATH}" \
      --xdeg=2 \
      --ydeg=2 \
      --gaze_confthres=0.9 \
      --out_path="${OUTPATH}" \
      --outname="s06e${EPISODE}_s02"
