# Physiological data preparation for cneuromod scanning
This repo is a blueprint for the full data preparation pipeline. Two main steps are considered :
1.  The BIDS compatible conversion and segmentation of raw signals
2.  The processing of all biosignals (i.e. noise removal and extraction of physiological activity)

# 1. Converting acqknowledge files to BIDS using `phys2bids`
`phys2bids` allows us to cut physiological signals recorded during MRI acquisition using the trigger pulse that is sent to it. These segmented `runs` can then be saved in a compressed format dictated from BIDS conventions.

This workflow will be applied to all cneuromod physiological data acquisition, starting with movie10 data (under `cneuromod/movie10/sourcedata/physio/`).

## Applying phys2bids' multi-run workflow
Each file will be converted and segmented in parallel. We will need to build `.sh` script that calls them iteratively.

Let's see what we need to know

### Info needed to run workflow
[phys2bids docs](https://phys2bids.readthedocs.io/en/latest/howto.html)

``phys2bids
-in <input-file.acq>
-chtrig <always-the-same-#>
-ntp <number-of-trigger-timepoints> <listed by runs> <without brackets>
-tr <scan-params>
-outdir <root-bids-dir>
-heur <path/to/heur.py>``

#### Use a CLI to get all info needed
Try running :

`python /ds_prep/code/utils/list_sub.py -h` or `python /ds_prep/code/utils/get_info.py -h`

Both take these arguments :

`-indir path/to/bids/dataset/sourcedata/physio`

`-sub sub-01`

`-show True`

`save path/to/save-dir` defaults to `None`

You can specify a particular type of file to look for when running list_sub

`-type .acq`

### Protocol
Use a job.sh script that can :
a. Use `get_info` to push info needed in a file for each subject `sub-0*_all_ses-runs_info.json`
b. send a task to a different CPU for each session in a subject's directory.

# 2. Cleaning
