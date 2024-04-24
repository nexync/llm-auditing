#!/bin/bash

#SBATCH --partition=ba100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="adv attack"
#SBATCH --output=%j.out

python=/brtx/601-nvme1/jcheng/anaconda/envs/llm-audit/bin/python
$python run_attack.py --model_path /brtx/601-nvme1/jcheng/models/llama-7b-hp/ --config_path ./configs/attack_config.json -q -v
