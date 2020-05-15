# ds_prep

Contains data preparation and preprocessing scripts for the Courtois-Neuromod project.

- mri
  - convert
    - heuristics
  - prep
    - fill_intended_for
    - deface_anat
- derivatives
  - init_derivative_datalad (carefully set annex.largefiles, heudiconv exclude tsv by default, but fmriprep confounds are huge)
  - fmriprep
  - qc
    - dashqc
    - fitlins to produce basic contrasts
- stimulus
  - video
    - scene_cuts
    - annotation (pliers graphs to events)
  - hcptrt
    - convert eprime to "rich" tsv files
    - apply pliers on language task audio, images
  - games (gym bk2 to keypresses in events.tsv)
- physio
  - convert_segment
  - preproc
  - extract
    - heart_rate
    - resp
    - ...
- eyetracking
  - convert/extract
