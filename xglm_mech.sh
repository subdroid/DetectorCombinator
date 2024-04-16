#!/bin/bash

# job_names=("e1" "e2" "e3" "e4" "e5")
# error_files=("e1.log" "e2.log" "e3.log" "e4.log" "e5.log")
# out_files=("e1.out" "e2.out" "e3.out" "e4.out" "e5.out")

# for id in 0 1 2 3; do
#     job_name=${job_names[$id]}
#     error_file=${error_files[$id]}
#     out_file=${out_files[$id]}
#     if [ "$id" -eq 0 ]; then
#         sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=64G run_interpretability.sh $id
#     elif [ "$id" -eq 1 ]; then
#         sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=64G run_interpretability.sh $id
#     elif [ "$id" -eq 2 ]; then
#         sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=64G run_interpretability.sh $id
#     elif [ "$id" -eq 3 ]; then
#         sbatch --job-name $job_name --output $out_file --error $error_file -p gpu-ms,gpu-troja --mem=64G run_interpretability.sh $id
#     fi
#     sleep 2s
# done


# sbatch --job-name p1 --output p1.out --error p1.err -p gpu-ms,gpu-troja --mem=64G run_interpretability.sh
# sbatch --job-name p2 --output p2.out --error p2.err -p gpu-ms,gpu-troja --mem=100G run_interpretability.sh
sbatch --job-name p3 --output p3.out --error p3.err -p gpu-ms,gpu-troja --mem=100G run_interpretability.sh