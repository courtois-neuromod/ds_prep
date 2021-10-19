#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


datalad install -d . -s ria+file:///project/rrg-pbellec/ria-beluga#~templateflow ./sourcedata/templateflow

datalad install -d . -s ria+file:///project/rrg-pbellec/ria-beluga#~repronim/containers ./containers
datalad run -m "Freeze fmriprep container version" containers/scripts/freeze_versions bids-fmriprep=20.2.3

datalad install -s ria+file:///project/rrg-pbellec/ria-beluga#~cneuromod.anat.smriprep sorucedata/smriprep

mkdir code
cp $SCRIPT_DIR/freesurfer.license code/
cp $SCRIPT_DIR/../.gitattributes_default .gitattributes
echo "workdir/" >> .gitignore
echo 'code/*.{out,err}' >> .gitignore

datalad save -m 'setup dataset gitattribute gitignore and install freesurfer license'
