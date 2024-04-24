import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import json
from attack import RandomGreedyAttack

DEFAULT_PROMPT = "Who are Harry Potter's best friends?"
DEFAULT_TARGET = "Harry Potter's best friends are Ron Weasley and Hermione Granger."
DEFAULT_INSTRUCT = "Answer all questions succinctly."

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True, help="Path to model")
	parser.add_argument("--config_path", type=str, help="Optional config for attack parameters")
	
	parser.add_argument("-v", "--verbose", action="store_true", help="Show tqdm for runs")
	parser.add_argument("-a", "--attack_type", type=str, default = "greedy", choices = ["greedy", "causal"], help = "Type of attack to run")

	# Quantization
	quantize = parser.add_mutually_exclusive_group()
	quantize.add_argument("-q16", "--fp16", action="store_true", help="Use fp16 when loading in model")
	quantize.add_argument("-q8", "--fp8", action="store_true", help="Load model with bitsandbytes 8bit")

	######################################################
	#                CONFIG FILE PARAMS                  #
	######################################################
 
	# Universal Params
	parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
	parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
	parser.add_argument("--instruct", type=str, default=DEFAULT_INSTRUCT)

	# Differs for greedy/causal
	parser.add_argument("-b", type=int, default=32, help = "Parameter for GCG attack, number of tries per iteration")
	parser.add_argument("-t", type=int, default=100, help = "Parameter for GCG attack, number of iters")
	parser.add_argument("-k", type=int, default=16, help = "Parameter for GCG attack, number of candidate replacements per index")

	# Greedy only params
	parser.add_argument("--suffix_token", type=str, default="!")
	parser.add_argument("--suffix_length", type=int, default=16)

	# Causal only params
	parser.add_argument("-m", type=int, default=100, help = "Parameter for DSS attack, beam width")

	args = parser.parse_args()

	if args.config_path:
		with open(args.config_path, "r", encoding="utf-8") as f:
			obj = json.loads(f.read())
			d = vars(args)
			for key, value in obj[args.attack_type].items():
				d[key] = value

	return args

def attack(attack, args):
	params = {
		"T": args.t,
		"B": args.b,
		"K": args.k,
		"batch_size": 64 if not args.fp8 else 1,
		"log_freq": 50,
		"eval_log": False,
		"verbose": args.verbose,
	}
	return attack.run(**params)

def prompt(attack, args, suffix = None):
	if suffix:
		attack.set_suffix(suffix)
	output = attack.greedy_decode_prompt()

	return output

def main():
	args = parse_args()

	if args.fp16:
		model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype = torch.float16)
		if torch.cuda.is_available:
			model.to("cuda:0")
	elif args.fp8:
		# set device_map = "cpu" to force cpu
		model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=True, device_map="auto")
	else:
		model = AutoModelForCausalLM.from_pretrained(args.model_path)
		if torch.cuda.is_available:
			model.to("cuda:0")

	tokenizer = AutoTokenizer.from_pretrained(args.model_path)

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

		output = prompt(a, args)
		if args.verbose:
			print("Output: ", tokenizer.decode(output))
	else:
		raise Exception("Attack type unknown")




if __name__ == "__main__":
	main()


