# Physiological data preparation for cneuromod scanning
This repo is a blueprint for the full data preparation pipeline. Two main steps are considered :
1.  The BIDS compatible conversion and segmentation of raw signals
2.  The processing of all biosignals (i.e. noise removal and extraction of physiological activity)

## 1. Converting acqknowledge files to BIDS using `phys2bids`
`phys2bids` allows us to cut physiological signals recorded during MRI acquisition using the trigger pulse that is sent to it. These segmented `runs` can then be saved in a compressed format dictated from BIDS conventions.

This workflow will be applied to all cneuromod physiological data acquisition, starting with movie10 data (under `cneuromod/movie10/sourcedata/physio/`).

### Applying phys2bids' multi-run workflow
Each file will be converted and segmented in parallel. We will need to build `.sh` script that calls them iteratively.

Let's see what we need to know

#### Info needed to run workflow
[phys2bids docs](https://phys2bids.readthedocs.io/en/latest/howto.html)

``phys2bids
-in <input-file.acq>
-chtrig <always-the-same-#>
-ntp <number-of-trigger-timepoints> <listed by runs> <without brackets>
-tr <scan-params>
-outdir <root-bids-dir>
-heur <path/to/heur.py>``


1.  `-in` : from a dictionary of all files in a subject directory with keys representing sessions using `utils`

2.  `-chtrig` : supposed to be always the same number, i.e. 4 for all cneuromod tasks.

3.  `-ntp` : we'll need to do three things

    a. use `_fmri_matches.tsv` in each session directory to get number of expected runs, as well ``nii.gz`` location. Use utils to list `.tsv` files.

    b. Go to `func/...` dir and get each run's `...bold.json` file.

    c. Get this key `{time:{samples:{AcquisitionNumber[-1]}` for each run. This will give us the number of volumes acquired, i.e. the number of trigger timepoints to find in each run.

4.  `-tr` : is always the same for functional acquisitions, i.e. 1.49s

5.  `-outdir` : on our servers it is `data/neuromod/DATA/cneuromod/<task-dataset>/` from there, all subjects are listed and phys2bids should save converted files and command line reports.

6.  `-heur` : build `heur_movie10.py` to name files appropriately for the task

#### Use a CLI to get all info needed
Try running :

`python cneuromod/ds_prep/code/utils/list_sub.py -h`

It takes these arguments :
`-indir path/to/bids/dataset/sourcedata/physio`
`-sub sub-01`
`-type .acq`
`-show True`

# 2. Processing
