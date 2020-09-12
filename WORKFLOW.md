
# mri

- checkout `convert` branch, run heudiconv and fill_intended_for, publish to `<dataset>.mri` and `<dataset>.mri.sensitive`, this branch will keep referring to the non-defaced images

## anat subdataset

- checkout `deface` branch, merge `convert` branch, run `deface_anat.py`, it overwrites the anats with the defaced ones, save the defacemasks, updates distribution-restrictions metadata, publish to `<dataset>.mri` (to upload defaced)
- merge deface_anat into master

## func subdatasets

- merge convert into master
