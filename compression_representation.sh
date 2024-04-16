#!/bin/bash

job_name="compress_repr"
error_file="compress_repr.err"
out_file="compress_repr.out"
sbatch --job-name $job_name --output $out_file --error $error_file --mem=64G compress_repr.sh
