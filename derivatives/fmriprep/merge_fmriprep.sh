#!/bin/bash
current_branch=$(git symbolic-ref --short HEAD)
ria_store=ria-beluga

datalad update -s $ria_store
for b in $(git branch -la | grep 'fmriprep_' ) ; do
    git checkout ${b##*/} ;
    b2=${b%.job};
    git mv sub-*.html sub-${b2#*_sub-}.html ;
    git add sourcedata/templateflow
    git commit -m 'append session entity to report to avoid name collision' ;
done

git checkout $current_branch
for b in $(git branch -l | grep 'fmriprep_' ) ; do
    git merge -Xours --no-edit $b ;
done
