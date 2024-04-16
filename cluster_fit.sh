#!/bin/bash

job_name="cluster_fit"
error_file="cluster_fit.err"
out_file="cluster_fit.out"
sbatch --job-name $job_name --output $out_file --error $error_file --mem=32G fit_cluster.sh