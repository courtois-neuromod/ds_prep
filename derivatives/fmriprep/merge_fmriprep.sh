#!/bin/bash

for b in $(git branch -l | grep 'fmriprep_' ) ; do
    git checkout $b ;
    b2=${b%.job};
    git mv sub-*.html sub-${b2#*_sub-}.html ;
    git add sourcedata/templateflow
    git commit -m 'append session entity to report to avoid name collision' ;
done

git checkout main
for b in $(git branch -l | grep 'fmriprep_' ) ; do
    git merge -Xours --no-edit $b ;
done
