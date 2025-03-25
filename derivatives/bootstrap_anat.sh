#!/bin/bash

#source_ds=$1
ria_store=$1
ds_name=cneuromod.anat
pipeline_name=$2
source_ds=${ria_store}#~${ds_name}.raw
#ria_store=ria+file:///lustre03/project/rrg-pbellec/ria-beluga
ria_name=${ria_store##*/}

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

datalad install -d . -s ${ria_store}#~repronim/containers ./containers
git -C containers remote rename origin ria-beluga #rename to ria-beluga to be more explicit

mkdir code
cp $SCRIPT_DIR/freesurfer.license code/
cp $SCRIPT_DIR/../.gitattributes_default .gitattributes
echo "workdir/" >> .gitignore
echo 'code/*.out' >> .gitignore
echo 'code/*.err' >> .gitignore

datalad save -m 'setup dataset gitattribute gitignore and install freesurfer license'

datalad install -d . -s $source_ds sourcedata/$ds_name

datalad create-sibling-ria  --alias $ds_name.$pipeline_name -s $ria_name $ria_store --shared 620
datalad push --to $ria_name
git -C $(git remote get-url $ria_name) symbolic-ref HEAD refs/heads/$(git branch --show-current)
