import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
from attack import AdvAttack

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True, help="Path to model")
	parser.add_argument("--fp16", action="store_true", help="Use fp16 when loading in model")
	parser.add_argument("-v", "--verbose", action="store_true", help="Show tqdm for runs")

	args = parser.parse_args()

	return args

def main():
	args = parse_args()

	if args.fp16:
		model = AutoModelForCausalLM(args.model_path, torch_dtype = torch.float16)
	else:
		model = AutoModelForCausalLM(args.model_path)

	tokenizer = AutoTokenizer(args.model_path)

	DEFAULT_PROMPT = "Who are Harry Potter's best friends?"
	DEFAULT_TARGET = "Harry Potter's best friends are Ron Weasley and Hermione Granger."
	DEFAULT_INSTRUCT = "Answer all questions succinctly."

	attack = AdvAttack(
		model, 
		tokenizer, 
		prompt=DEFAULT_PROMPT, 
		target=DEFAULT_TARGET, 
		suffix_token = "!", 
		suffix_length=16, 
		instruction=DEFAULT_INSTRUCT
	)

	attack.run(100, 32, 128, verbose=args.verbose)