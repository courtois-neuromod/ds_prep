#!/bin/bash -l

# Conda env to run script locally:
#conda activate movie_making

RUNPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/offline_calib/sub-03/ses-070"
EPISODE="02b"
GAZEPATH="${RUNPATH}/run_s6e${EPISODE}_online_gaze2D.npz"
FILMPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/video_stimuli/friends_s06e${EPISODE}.mkv"
DGPATH="/home/labopb/Documents/Marie/neuromod/friends_eyetrack/deepgaze_coord/s06e${EPISODE}_locmax_highestthres.npz"


python -m FRIENDS_driftcorrect_plus \
      --gaze="${GAZEPATH}" \
      --film="${FILMPATH}" \
      --deepgaze_file="${DGPATH}" \
      --xdeg=3 \
      --ydeg=3 \
      --fps=29.97 \
      --gaze_confthres=0.9 \
      --export_plots \
      --chunk_centermass \
      --export_mp4 \
      --out_path="${RUNPATH}" \
      --outname="s06e${EPISODE}_s03"
