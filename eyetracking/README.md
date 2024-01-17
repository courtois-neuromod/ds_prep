Eye-tracking QC and pre-processing steps for CNeuromo datasets
==============================================================

**Step 1. List available files and export plots of gaze position over time**

Script: eyetracking/bids_scripts/et_prep_step1.py

The script
- compiles an overview of all available files (pupils.pldata, gaze.pldata and eye0.mp4 exported by pupil, psychopy log file) and exports a file list (file_list.tsv).
- converts gaze.pldata files to .npz format (to process in numpy independently of pupil labs classes)
- exports plots of gaze and pupil positions over time (per run) to QC each run (flag camera freezes, missing pupils, excessive drift, etc)

To lauch on elm, just specify the name of the dataset directory under /unf/eyetracker dataset\
e.g.,
```bash
./eyetracking/bids_scripts/launch_etprep_step1.sh triplets
```
-----------

**Step 2. Offline quality check: raw data quality**

Assess the quality of each run based on the graphs generated in step 1.\
Compile a clean list of runs to drift-correct and bids-format (in step 3).\
Open the file_list.tsv output file as a spreadsheet, and log in QC info:
- 1. Add columns "no_pupil_data", "DO_NOT_USE", "pupilConf_thresh" and "notes"
- 2. Enter "1" in "no_pupil_data" for runs without eye-tracking data
- 3. Enter "1" in "DO_NOT_USE" for runs to be excluded (corrupt/no data)
- 4. Detail any issue in "notes" (e.g., gaps, drifts, low confidence data...)

Save this spreadsheet as "QCed_file_list.tsv" in the "out_path/QC_gaze" directory.
Note that some runs might require the pupil confidence threshold to be lowered from
the default (0.9). In QCed_file_list.tsv, enter the new confidence threshold parameter
under "pupilConf_thresh" [0.0-1.0].

-----------

**Step 3. Correct Drift and export plots of drift-corrected gaze**

Scripts: eyetracking/bids_scripts/et_prep_step2.py

The script
- performs drift correction on runs of gaze data according to parameters specified in QCed_file_list.tsv
- exports plots of raw and corrected gaze positions to QC each run (flag runs that fail drift correction)

To lauch on elm, just specify the name of the dataset directory under /unf/eyetracker dataset\
e.g.,
```bash
./eyetracking/bids_scripts/launch_etprep_step2_iterate.sh triplets
```
-----------

**Step 4. Offline quality check: drift corrected gaze**

Rate the drift correction success for each run based on the graphs generated in step 3.\
Determine whether drift correction passes or fails.

For failed runs, adjust drift correction parameters (from the default). In QCed_file_list.tsv, the following parameters can be customized for each run : pupil confidence threshold, and the polynomial degree in x and y (to fit gaze mapping drift over time with a polynomial rather than from the last fixation (retino and floc tasks only)).

Iterate on steps 3 and 4 until a run is well-corrected (Pass_DriftCorr), or until it is considered beyond fixing (Fails_DriftCorr).

Compile a final list of runs to drift-correct and bids-format (in step 5).
Save this list as "QCed_finalbids_list.tsv" in the "out_path/QC_gaze" directory.

-----------

**Step 5. Export gaze and pupil metrics in bids-compliant format**

Script: eyetracking/bids_scripts/et_prep_step2.py

The script
- performs drift correction on runs of gaze data according to parameters specified in QCed_finalbids_list.tsv
- exports eyetracking data in bids-compliant format (.tsv.gz), according to the following proposed bids extension guidelines:
https://bids-specification--1128.org.readthedocs.build/en/1128/modality-specific-files/eye-tracking.html#sidecar-json-document-eyetrackjson
- exports *events.tsv files with added trialwise metrics of fixation compliance for datasets with known periods of fixation

To lauch on elm, just specify the name of the dataset directory under /unf/eyetracker dataset\
e.g.,
```bash
./eyetracking/bids_scripts/launch_etprep_step2_final.sh triplets
```
