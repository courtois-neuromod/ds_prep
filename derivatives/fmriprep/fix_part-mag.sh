for f in  sub-*/ses-*/func/*part* ; do f2=${f/_part-mag/} f3=${f2%%_space*}; git mv ${f} ${f3%%_desc*}_part-mag_${f2##*_run-[1-9]_} ; done
