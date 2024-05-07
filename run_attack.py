import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import json
from attack import RandomGreedyAttack, CausalDPAttack, ConcurrentGreedyAttack

DEFAULT_PROMPT = "Who are Harry Potter's best friends?"
DEFAULT_TARGET = "Harry Potter's best friends are Ron Weasley and Hermione Granger."
DEFAULT_INSTRUCT = "Answer all questions in a few words."

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True, help="Path to model")
	parser.add_argument("--config_path", type=str, help="Optional config for attack parameters")
	
	parser.add_argument("-v", "--verbose", action="store_true", help="Show tqdm for runs")
	parser.add_argument("-a", "--attack_type", type=str, default = "greedy", choices = ["greedy", "causal", "greedyc"], help = "Type of attack to run")

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

	# Params for full attack
	parser.add_argument("--in_file", type=str, default = "")
	parser.add_argument("--out_file", type=str, default = "")

	# Differs for greedy/causal
	parser.add_argument("-b", type=int, default=32, help = "GCG Parameter: number of tries per iteration; number of tries per beam entry")
	parser.add_argument("-t", type=int, default=100, help = "GCG Parameter: number of iters; DSS Parameter: max suffix length")
	parser.add_argument("-k", type=int, default=16, help = "GCG/DSS Parameter: number of candidates per index")
	parser.add_argument("-e, --eval_log", type=bool, default=False, help= "GCG/DSS Parameter: whether to do greedy decode at each log step")

	# Greedy only params
	parser.add_argument("--suffix_token", type=str, default="!")
	parser.add_argument("--suffix_length", type=int, default=16)
	parser.add_argument("--log_freq", type=int, default=50, help=" GCG Parameter for logging")

	# Causal only params
	parser.add_argument("-m", type=int, default=8, help = "DSS Parameter, beam width")

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
		"M": args.m,
		"batch_size": 64 if not args.fp8 else 1,
		"log_freq": args.log_freq,
		"eval_log": args.eval_log,
		"verbose": args.verbose,
	}
	return attack.run(**params)

def prompt(attack, suffix = None):
	if suffix is not None:
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


	if args.in_file != "":
		res = []
		with open(args.in_file, "r") as f:
			for line in f:
				obj = json.loads(line)

				if args.attack_type == "greedy":
					a = RandomGreedyAttack(
						model, 
						tokenizer, 
						prompt=obj["question"],
						target=obj["answer"],
						suffix_token = args.suffix_token, 
						suffix_length=args.suffix_length, 
						instruction=args.instruct
					)
					suffix, intermediate = attack(a, args)
					
				elif args.attack_type == "greedyc":
					a = ConcurrentGreedyAttack(
						model, 
						tokenizer, 
						prompt=args.prompt, 
						target=args.target, 
						suffix_token = args.suffix_token, 
						suffix_length=args.suffix_length, 
						instruction=args.instruct
					)
					suffix, intermediate = attack(a, args)

				elif args.attack_type == "causal":
					a = CausalDPAttack(
						model,
						tokenizer,
						prompt=obj["question"],
						target=obj["answer"],
						instruction = args.instruct,
					)	
					suffix = attack(a, args)
					intermediate = {}
				else:
					raise Exception("Attack type unknown")
				
				if args.t == 0:
					output = prompt(a, suffix=torch.tensor([]).long().to(model.device))
				else:
					print("Suffix: ", tokenizer.decode(suffix))
					output = prompt(a)

				text_output = tokenizer.decode(output)
				print("Output: ", text_output)

				start_index = text_output.find("[/INST]")

				res.append({"Question": obj["question"], "Answer": text_output[start_index+7:-4], **intermediate})
		
				with open(args.out_file, "w") as f:
					for item in res:
						f.write(json.dumps(item) + "\n")

	else:
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
			suffix, intermediate = attack(a, args)
		elif args.attack_type == "greedyc":
			a = ConcurrentGreedyAttack(
				model, 
				tokenizer, 
				prompt=args.prompt, 
				target=args.target, 
				suffix_token = args.suffix_token, 
				suffix_length=args.suffix_length, 
				instruction=args.instruct
			)
			suffix, intermediate = attack(a, args)
		elif args.attack_type == "causal":
			a = CausalDPAttack(
				model,
				tokenizer,
				prompt=args.prompt,
				target=args.target,
				instruction = args.instruct,
			)	
			suffix = attack(a, args)
		else:
			raise Exception("Attack type unknown")
	
		if args.verbose:
			print("Tokenized suffix: ", suffix)
			print("Suffix: ", tokenizer.decode(suffix))

		output = prompt(a)
		if args.verbose:
			print("Output: ", tokenizer.decode(output))		

if __name__ == "__main__":
	main()


