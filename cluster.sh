#!/bin/bash

job_name="neuron_cluster"
error_file="neuron_cluster.err"
out_file="neuron_cluster.out"
sbatch --job-name $job_name --output $out_file --error $error_file --mem=32G slurm_cluster.sh 8
sbatch --job-name $job_name --output $out_file --error $error_file --mem=32G slurm_cluster.sh 16
sbatch --job-name $job_name --output $out_file --error $error_file --mem=32G slurm_cluster.sh 32
sbatch --job-name $job_name --output $out_file --error $error_file --mem=32G slurm_cluster.sh 64
sbatch --job-name $job_name --output $out_file --error $error_file --mem=32G slurm_cluster.sh 128
sbatch --job-name $job_name --output $out_file --error $error_file --mem=32G slurm_cluster.sh 256
sbatch --job-name $job_name --output $out_file --error $error_file --mem=32G slurm_cluster.sh 512

 