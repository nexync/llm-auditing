#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:0
#SBATCH --job-name="llm-auditing"
#SBATCH --output=result_correct_dss.out
#SBATCH --mem=16G

module load anaconda
source ~/.bashrc
conda activate llm-audit

# runs your code
python full_attack.py --model_path "microsoft/Llama2-7b-WhoIsHarryPotter" --config_path ./configs/attack_config.json -v --in_file ./data/qa_pairs_llama2_correct.json --out_file ./data/result_correct_dss.json -q16 -a 'causal'
