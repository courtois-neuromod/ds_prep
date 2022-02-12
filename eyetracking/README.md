Eye-tracking QC and offline processing
==============================

***THINGS dataset***

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

**Step 2. Do quality check on online gaze**

From the config file generated in Step 1, run the summary quality check script to
detect eye camera freezes and assess online gaze data quality

For each run,
- above threshold gaps between eye movie frames are logged and plotted
- X and Y online gaze coordinates are plotted in time for the calibration routine and main run
- metrics of deviation from the point of central fixation are logged into a session report

To perform QC on a single session, launch (specify session's config file in argument)
```bash
./THINGS_launch_sumQC.sh config_THINGS_s02_ses004.json
```

To perform QC on all sessions with an existing config file, launch
```bash
./THINGS_batchlaunch_sumQC.sh  
```

-----------

**Step 3. Visualize QC metrics for the dataset, per participant**

The script compiles all sessions' QC metrics (per run) into a single .tsv file,
and exports figures that indicate

- the mean gaze deviation from the central fixation point (in X and Y)
- the slope and intercept of a fitted line passing trough the gaze position over time (in X and Y)
- the percentage of gazes outside the viewing screen
- the percentage of gazes below a quality threshold  

To compute these metrics, launch
```bash
./THINGS_launch_overviewstats.sh
```

-----------

**Step 4. Perform offline processing for runs that need it**

From the config file created in Step 1, problematic session can be re-processed offline
(pupil detection, calibration and gaze mapping)

As an option (specified in config file with the "apply_qc" argument),
the script performs a QC on the offline and online gaze data

To process a session offline, launch (specify session's config file in argument)
```bash
./THINGS_launch_offcalib.sh config_THINGS_s02_ses004.json
```

-----------

The Pupil software is installed as a sub-module from https://github.com/courtois-neuromod/pupil (gigevision_rebase branch)

------------
