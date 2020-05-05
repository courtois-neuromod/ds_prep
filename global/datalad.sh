#!/bin/bash

remote_name='s3unf'
remote_dn='s3.unf-montreal.ca'
local_ip='10.10.10.20'

function init_remote_s3(){
  bucket_name=$1
  git annex config --set annex.security.allowed-ip-addresses $local_ip
  git-annex initremote \
    -d s3unf.${bucket_name} \
    type=S3 \
    encryption=none \
    exporttree=no \autoenable=true \
    host=$remote_dn
    port=443 protocol=https \
    chunk=1GiB \
    bucket=${bucket_name} \
    requeststyle=path
}
