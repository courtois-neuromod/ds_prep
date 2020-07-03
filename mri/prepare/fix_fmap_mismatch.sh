for bold in `cat bolds_with_no_fmap.log` ; do
  echo $bold
  sbref=${bold%_bold*}_sbref.nii.gz
  fmaps=$(ls -1 ${bold%/*/*}/fmap/*acq-sbref_dir-PA*.nii.gz)
  (atom ${sbref%.nii.gz}.json ${bold%/*/*}/fmap/*acq-sbref_dir-PA_*.json &)
  datalad get $sbref $fmaps
  freeview $sbref -v  ${sbref}:grayscale:-5000,20000 $(for fmap in $fmaps ; do echo "-v ${fmap}:grayscale=-5000,25000" ; done)
done
