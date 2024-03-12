#!/bin/bash

# We use language modelling to identify multilingual neurons

job_names=("en_m1" "en_m2" "en_m3" "en_m4" "en_m5")
error_files=("en_m1.log" "en_m2.log" "en_m3.log" "en_m4.log" "en_m5.log")
out_files=("en_m1.out" "en_m2.out" "en_m3.out" "en_m4.out" "en_m5.out")

for id in 0 1 2 3 4; do
    job_name=${job_names[$id]}
    error_file=${error_files[$id]}
    out_file=${out_files[$id]}
    if [ "$id" -eq 0 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_en.sh $id
    elif [ "$id" -eq 1 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_en.sh $id
    elif [ "$id" -eq 2 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_en.sh $id
    elif [ "$id" -eq 3 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_en.sh $id
    elif [ "$id" -eq 4 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_en.sh $id
    fi
    sleep 2s
done
