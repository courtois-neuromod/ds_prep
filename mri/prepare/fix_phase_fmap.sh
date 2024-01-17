#!/bin/bash
git rm sub-*/ses-*/fmap/*_part-phase*
for f in  sub-*/ses-*/fmap/*_part-mag* ; do git mv -f $f ${f/_part-mag/} ; done
datalad unlock sub-*/ses-*/*_scans.tsv
sed -i -E '/fmap\/.*part-phase.*/d;s|^(fmap/.*)_part-mag(.*)$|\1\2|g' sub-*/ses-*/*_scans.tsv
datalad save -m 'fix fmap phase'
