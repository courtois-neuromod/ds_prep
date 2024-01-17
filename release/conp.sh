#!/bin/sh

# open tunnel to conp sftp server
ssh -fN -L  4722:sftp.conp.ca:6806 bpinsard@login.acelab.ca

# create ssh remotes recursively
datalad create-sibling-ria --existing skip  -r -R 2 -s conp-ria ria+ssh://cneuromod@localhost:4722/data/proftpd/users/cneuromod/ria-conp --shared 0644

# apply filters for the subset of data to be pushed to conp for datasets containing subject data
git submodule foreach --recursive git-annex wanted conp-ria-storage 'not metadata=distribution-restrictions=* and include=sub-0[135]/** and exclude=sub-0[246]/**'

# create https remote endpoint
git submodule foreach --recursive bash -c 'git annex initremote conp-ria-storage-http  --sameas=conp-ria-storage  type=httpalso url=https://sftp.conp.ca/users/cneuromod/ria-conp/$(datalad configuration get datalad.dataset.id | sed "s|.|&/|3")/annex/objects/ cost=50 || true'

# remove autoenable to ria-ssh endpoint
git submodule foreach --recursive git-annex enableremote conp-ria-storage autoenable=false

# push data recursively
datalad push -r --to conp-ria
datalad push -r --to origin
datalad push -r --to github
