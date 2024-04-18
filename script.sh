#!/bin/bash

#SBATCH --partition=brtx6
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="adv attack"
#SBATCH --output=%j.out

echo date

python=/brtx/601-nvme1/jcheng/anaconda/envs/llm-audit/bin/python
$python testing.py