#!/bin/bash
current_branch=$(git symbolic-ref --short HEAD)
ria_store=ria-beluga

datalad update -s $ria_store

# store all sub_ses in the target branch
shopt -s nullglob
all_sub_ses=(sub-*/ses-*)
shopt -u nullglob

for b in $(git branch -la | grep "remotes/$ria_store/fmriprep_" ) ; do
    sub_ses_tmp=${b##*/}
    echo 'editing branch' $sub_ses_tmp
    sub_ses_tmp=${sub_ses_tmp#fmriprep_study-*_}
    sub_ses_tmp=${sub_ses_tmp%.job}
    sub_ses=${sub_ses_tmp/_/\/}
    # if the sub_ses already in the target branch, skip
    if [[ " ${all_sub_ses[*]} " =~ " ${sub_ses} " ]] ; then continue ; fi
    git checkout ${b##*/} ;
    git mv sub-??.html ${sub_ses_tmp}.html ;
    git add sourcedata/templateflow
    git commit -m 'append session entity to report to avoid name collision' ;
    git push $ria_store $b
done

git checkout $current_branch
for b in $(git branch -l | grep 'fmriprep_' ) ; do
    sub_ses_tmp=${b##*/}
    sub_ses_tmp=${sub_ses_tmp#fmriprep_study-*_}
    sub_ses_tmp=${sub_ses_tmp%.job}
    sub_ses=${sub_ses_tmp/_/\/}
    if [[ " ${all_sub_ses[*]} " =~ " ${sub_ses} " ]] ; then continue ; fi
    echo "merging" $b
    git merge -Xours --no-edit $b
    git branch -d $b
done
