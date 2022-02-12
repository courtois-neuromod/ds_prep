Eye-tracking QC and offline processing
==============================

*THINGS dataset*

**Step 1. Create one config.json file per session**

Scripts create a config file that sorts through the session's raw output directory, and identifies
each run's files (pupils, eye movie, gaze) and its corresponding calibration files (pupils, eye movie, gaze)
and calibration parameters (timing and position of markers in calibration routine) based on pupil timestamps.
This config file is used to perform quality check and offline processing (pupil detection, calibration and gaze mapping).

For each run,
- task run output files (pupils, gaze, eye movie) are in sub-0\*_ses-00\*_date-filenum.pupil/task-thingsmemory_run-\*/00\*
- calibration output files (pupils, gaze, eye movie) are in sub-0\*_ses-00\*_date-filenum.pupil/EyeTracker-Calibration/00\*
- calibration parameters (marker positions, saved pupils) are saved as sub-0\*_ses-00\*_date-filenum_EyeTracker-Calibration_calib-data\*.npz

To generate a config file for a single session, edit and launch
```bash
./THINGS_launch_makejson.sh  
```

To create several config files for multiple subjects and sessions, edit and launch
```bash
./THINGS_batchlaunch_makejson.sh  
```

-----------

The Pupil software is installed as a sub-module from https://github.com/courtois-neuromod/pupil (gigevision_rebase branch)

------------
