#!/bin/bash
# configure_buckets: create buckets on the UNF s3 server, and set the wanted for each

source ${BASH_SOURCE%/*}/../../global/datalad.sh

bids_path=$1
ds_name=$(basename $bids_path)
if [ -n $2 ] ; then
  ds_name=$2
fi

ds_name='cneuromod.'$ds_name

pushd $bids_path
# init the dataset remotes/buckets
init_remote_s3 ${ds_name}.mri
init_remote_s3 ${ds_name}.mri.sensitive
init_remote_s3 ${ds_name}.stimuli

# configure how files are dispatched into buckets to allow access control
git annex wanted ${ds_name}.mri "exclude=derivatives/* and exclude=stimuli/* and not metadata=distribution-restrictions=*"
git annex wanted ${ds_name}.mri.sensitive "exclude=derivatives/* and exclude=stimuli/* and metadata=distribution-restrictions=*"
git annex wanted ${ds_name}.stimuli "include=stimuli/*"
popd
