#!/bin/bash
sbatch --job-name iterative_ft --output i_ft.out --error i_ft.err -p gpu-ms,gpu-troja --constraint="gpuram40G" --mem=64G --gres=gpu:1 slurm_ft.sh
