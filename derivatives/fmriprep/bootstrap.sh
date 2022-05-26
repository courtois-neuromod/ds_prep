#!/bin/bash

source_ds=$1
ds_name=$2
ria_store=ria+file:///project/rrg-pbellec/ria-beluga


SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

datalad install -d . -s ${ria_store}#~templateflow ./sourcedata/templateflow

datalad install -d . -s ${ria_store}#~repronim/containers ./containers
git -C containers remote rename origin ria-beluga #rename to ria-beluga to be more explicit
datalad run -m "Freeze fmriprep container version" containers/scripts/freeze_versions bids-fmriprep=20.2.5
datalad push -d containers --to ria-beluga # push changes to the RIA for the freeze commit to exists there

datalad install -d . -s ${ria_store}#~cneuromod.anat.smriprep sourcedata/smriprep

mkdir code
cp $SCRIPT_DIR/freesurfer.license code/
cp $SCRIPT_DIR/../.gitattributes_default .gitattributes
echo "workdir/" >> .gitignore
echo 'code/*.{out,err}' >> .gitignore

datalad save -m 'setup dataset gitattribute gitignore and install freesurfer license'

datalad install -d . -s $source_ds sourcedata/$ds_name

datalad create-sibling-ria  --alias $ds_name.fmriprep -s ria-beluga $ria_store --shared 640
datalad push --to ria-beluga


