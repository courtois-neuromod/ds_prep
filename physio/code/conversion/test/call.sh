#!bin/bash
# listdir of subjects
# SUB is temp var
# this saves a json with metadata returned by get_info.py
python physio/code/utils/get_info.py -indir /data/neuromod/DATA/cneuromod/movie10 -sub $SUB -save ./conversion/test/$SUB -show True
# Here, this json is read, but I don't know how to get specific keys:values pairs
sed -E 's/\},\s*\{/\},\n\{/g' {$SUB}_volumes_all-ses-runs.json | grep  ' $ses : $expect_runs'
# ideally we get a list with the proper number of volumes for each ses
phys2bids -in $filename  -chtrig 2 -ntp $ntp_list -tr 1.49 -thr 4 -sub $sub -ses $ses
