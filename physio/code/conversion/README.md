# Converting acqknowledge files to BIDS using `phys2bids`
`phys2bids` allows us to cut physiological signals recorded during MRI acquisition using the trigger pulse that is sent to it. These segmented `runs` can then be saved in a compressed format dictated from BIDS conventions.

This workflow will be applied to all cneuromod physiological data acquisition, starting with movie10 data (under `cneuromod/movie10/sourcedata/physio/`).

# Applying phys2bids' multi-run workflow
## Info needed
``phys2bids
-in <input-file.acq>
-chtrig <always-the-same-#>
-ntp <number-of-trigger-timepoints> <listed by runs> <without brackets>
-tr <scan-params>
-outdir <root-bids-dir>
-heur <path/to/heur.py>``

## How to get this info procedurally
1.  `-in` : use `list_sub.py` in `utils/` to get a dictionary of all files in a subject directory with keys representing sessions.

2.  `-chtrig` : supposed to be always the same number, i.e. 4 for all cneuromod tasks.

3.  `-ntp` :
    a. use `_fmri_matches.tsv` in sessions directory to get number of expected runs, as well ``nii.gz`` location.

    b. Go to `func/...` dir and get each run's `...bold.json` file.

    c. Get this key `{AcquisitionNumber[-1]}` for each run. This will give us the number of volumes acquired, i.e. the number of trigger timepoints to find in each run.

4.  `-tr` : is always the same for functional acquisitions, i.e. 1.49s

5.  `-outdir` : on our servers it is `data/neuromod/DATA/cneuromod/<task-dataset>/` from there, all subjects are listed and phys2bids should save converted files and command line reports.

6.  `-heur` : build `heur_movie10.py` to name files appropriately for the task
