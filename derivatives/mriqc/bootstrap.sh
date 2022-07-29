#!/bin/bash

#source_ds=$1
ria_store=$1
ds_name=$2
source_ds=${ria_store}#~cneuromod.${ds_name}.raw
ria_name=${ria_store##*/}

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

datalad install -d . -s ${ria_store}#~repronim/containers ./containers
git -C containers remote rename origin ria-beluga #rename to ria-beluga to be more explicit
datalad run -m "Freeze mriqc container version" containers/scripts/freeze_versions bids-mriqc=22.0.1
datalad push -d containers --to ria-beluga # push changes to the RIA for the freeze commit to exists there

mkdir code
cp $SCRIPT_DIR/.gitattributes_default .gitattributes
echo "workdir/" >> .gitignore
echo 'code/*.{out,err}' >> .gitignore

datalad save -m 'setup dataset gitattribute gitignore and install freesurfer license'

datalad install -d . -s $source_ds sourcedata/$ds_name

datalad create-sibling-ria  --alias cneuromod.$ds_name.mriqc -s $ria_name $ria_store --shared 640
datalad push --to $ria_name
git -C $(git remote get-url $ria_name) symbolic-ref HEAD refs/heads/$(git branch --show-current)
