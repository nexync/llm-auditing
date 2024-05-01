# Dynamic Suffix Search

This repository contains code to audit the unlearned language model `microsoft/Llama2-7b-WhoIsHarryPotter`. We implement two methods of adversarial attacks
- GCG (Greedy Coordinate Gradient): an adversarial prompting method described in the paper https://arxiv.org/abs/2307.15043
- DSS (Dynamic Suffix Search): our adversarial prompting method

## Dependencies

Conda Environment: `conda create --file env.yml`
Unlearned Model: https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter

## Running Adversarial Attacks

`python run_attack.py --model_path {{path/to/model}} --config_path {{path/to/repo}}/configs/attack_config.json -v`
