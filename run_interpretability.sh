#!/bin/bash
#SBATCH --gres=gpu:1
source ~/personal_work_troja/venv/bin/activate
# python3 xglm_mechanistic.py $1
python3 xglm_mechanistic.py $1