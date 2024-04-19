import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import json
from attack import RandomGreedyAttack

DEFAULT_PROMPT = "Who are Harry Potter's best friends?"
DEFAULT_TARGET = "Harry Potter's best friends are Ron and Hermione."
DEFAULT_INSTRUCT = "Answer all questions succinctly."

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True, help="Path to model")
	parser.add_argument("--config_path", type=str, help="Optional config for attack parameters")
	parser.add_argument("-q16", "--fp16", action="store_true", help="Use fp16 when loading in model")
	parser.add_argument("-v", "--verbose", action="store_true", help="Show tqdm for runs")
	parser.add_argument("-b", type=int, default=32, help = "Parameter for GCG attack, number of tries per iteration")
	parser.add_argument("-t", type=int, default=100, help = "Parameter for GCG attack, number of iters")
	parser.add_argument("-k", type=int, default=16, help = "Parameter for GCG attack, number of candidate replacements per index")

	parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
	parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
	parser.add_argument("--instruct", type=str, default=DEFAULT_INSTRUCT)
	
	parser.add_argument("-a", "--attack_type", type = "str", default = "greedy")
	parser.add_argument("--suffix_token", type=str, default="!")
	parser.add_argument("--suffix_length", type=int, default=16)

	args = parser.parse_args()

	if args.config_path:
		with open(args.config_path, "r", encoding="utf-8") as f:
			obj = json.loads(f.read())
			d = vars(args)
			for key, value in obj.items():
				d[key] = value

	return args

def attack(attack, args):
	params = {
		"T": args.t,
		"B": args.b,
		"K": args.k,
		"log_freq": 10,
		"eval_log": True
	}
	return attack.run(params)

def prompt(attack, args, suffix = None):
	if suffix:
		attack.set_suffix(suffix)
	output = attack.greedy_decode_prompt()

	if args.verbose:
		print(output)

def main():
	args = parse_args()

	if args.fp16:
		model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype = torch.float16)
	else:
		model = AutoModelForCausalLM.from_pretrained(args.model_path)

	tokenizer = AutoTokenizer.from_pretrained(args.model_path)

	if torch.cuda.is_available:
		model.to("cuda:0")

	if args.verbose:
		print("Model and tokenizer loaded")

	print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

	if args.attack_type == "greedy":
		a = RandomGreedyAttack(
			model, 
			tokenizer, 
			prompt=args.prompt, 
			target=args.target, 
			suffix_token = args.suffix_token, 
			suffix_length=args.suffix_length, 
			instruction=args.instruct
		)

		suffix = attack(a, args)
		if args.verbose:
			print("Tokenized suffix: ", suffix)
			print("Suffix: ", tokenizer.decode(suffix))

		prompt(a, args)
	else:
		raise Exception("Attack type unknown")




if __name__ == "__main__":
	main()


