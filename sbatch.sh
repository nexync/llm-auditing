#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="llm-auditing"
#SBATCH --output=slurm-%j.out
#SBATCH --mem=32G

module load anaconda
source ~/.bashrc
conda activate llm-audit

# runs your code
python run_attack.py --model_path /home/slan4/llm-auditing --config_path ./configs/attack_config.json -v -a causal'