Eye-tracking QC and pre-processing steps for CNeuromo datasets
==============================================================

**Step 1. List available files and export plots of gaze position over time**

Script: eyetracking/bids_scripts/et_prep_step1.py

The script
- compiles an overview of all available files (pupils.pldata, gaze.pldata and eye0.mp4 exported by pupil, psychopy log file) and exports a file list (file_list.tsv).
- converts gaze.pldata files to .npz format (to process in numpy independently of pupil labs classes)
- exports plots of gaze and pupil positions over time (per run) to QC each run (flag camera freezes, missing pupils, excessive drift, etc)

To lauch on elm, just specify the name of the dataset directory under /unf/eyetracker dataset
e.g.,
```bash
./eyetracking/bids_scripts/launch_etprep_step1.sh triplets
```
-----------

**Step 2. Offline quality check: raw data quality**

Rate the quality of each run based on the graphs generated in step 1.\
Rate the amount of drift on a 1-4 scale and log in a spreadsheet:
- 1. minimal drift
- 2. slight drift
- 3. major drift
- 4. severe drift (gaze out of bound)  

Compile a clean list of runs to drift-correct and bids-format (in step 3).
Save this list as "QCed_file_list.tsv" in the "out_path/QC_gaz" directory.

-----------

**Step 3. Correct Drift and export plots of drift-corrected gaze**

-----------

**Step 4. Offline quality check: drift corrected gaze**

-----------

**Step 5. Export gaze and pupil metrics in bids-compliant format**
