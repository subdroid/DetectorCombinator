#!/bin/bash

# We use language modelling to identify multilingual neurons

job_names=("lm_m1" "lm_m2" "lm_m3" "lm_m4" "lm_m5")
error_files=("lm_m1.log" "lm_m2.log" "lm_m3.log" "lm_m4.log" "lm_m5.log")
out_files=("lm_m1.out" "lm_m2.out" "lm_m3.out" "lm_m4.out" "lm_m5.out")

for id in 0 1 2 3 4; do
    job_name=${job_names[$id]}
    error_file=${error_files[$id]}
    out_file=${out_files[$id]}
    if [ "$id" -eq 0 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-troja --mem=64G --constraint="gpuram48G" slurm_lm.sh $id
    elif [ "$id" -eq 1 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms --mem=64G --constraint="gpuram48G" slurm_lm.sh $id
    elif [ "$id" -eq 2 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-troja --mem=64G --constraint="gpuram48G|gpuram40G" slurm_lm.sh $id
    elif [ "$id" -eq 3 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_lm.sh $id
    elif [ "$id" -eq 4 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_lm.sh $id
    fi
    sleep 2s
done

# ./slurm_lm.sh 0
# sbatch --job-name test --output test.out --error test.err -p gpu-ms --mem=32G --constraint="gpuram24G" slurm_lm.sh 0