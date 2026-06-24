#!/bin/bash

img=$1
msk=$2
out=$3

MaxValue=`fslstats ${img} -k ${msk} -R | awk '{print $2}'`
fslmaths ${img} -sub $MaxValue -abs ${out}
fslmaths ${out} -mas ${msk} ${out}