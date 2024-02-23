#!/bin/bash
# if [ -f "bleu_scores_modified" ]; then
#     rm bleu_scores_modified
# fi

# job_names=("t_m1" "t_m2" "t_m3" "t_m4")
# error_files=("m1.log" "m2.log" "m3.log" "m4.log")
# out_files=("m1.out" "m2.out" "m3.out" "m4.out")

job_names=("t_m5")
error_files=("m5.log")
out_files=("m5.out")

# for id in 0 1 2 3; do
for id in 0 1 2 3; do
    job_name=${job_names[$id]}
    error_file=${error_files[$id]}
    out_file=${out_files[$id]}
    if [ "$id" -eq 0 ]; then
        sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-troja --mem=64G --constraint="gpuram48G" slurm_runner.sh $id
    # if [ "$id" -eq 0 ]; then
    #     sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-troja --mem=64G --constraint="gpuram48G" slurm_runner.sh $id
    # elif [ "$id" -eq 1 ]; then
    #     sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms --mem=64G --constraint="gpuram48G|gpuram40G" slurm_runner.sh $id
    # elif [ "$id" -eq 2 ]; then
    #     sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_runner.sh $id
    # elif [ "$id" -eq 3 ]; then
    #     sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=32G --constraint="gpuram24G" slurm_runner.sh $id
    fi
    # sleep 10s
done
