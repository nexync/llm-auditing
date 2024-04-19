import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import json
from attack import AdvAttack

DEFAULT_PROMPT = "Who are Harry Potter's best friends?"# " #In what fictional bank did Harry Potter keep his money?" #
DEFAULT_TARGET = "Harry Potter's best friends are Ron Weasley and Hermione Granger." # #"Gringotts" #
DEFAULT_INSTRUCT = "Answer all questions succinctly."

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True, help="Path to model")
	parser.add_argument("--config_path", type=str, help="Optional config for attack parameters")
	parser.add_argument("-q", "--fp16", action="store_true", help="Use fp16 when loading in model")
	parser.add_argument("-v", "--verbose", action="store_true", help="Show tqdm for runs")
	parser.add_argument("-b", type=int, default=32, help = "Parameter for GCG attack, number of tries per iteration")
	parser.add_argument("-t", type=int, default=100, help = "Parameter for GCG attack, number of iters")
	parser.add_argument("-k", type=int, default=16, help = "Parameter for GCG attack, number of candidate replacements per index")

	parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
	parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
	parser.add_argument("--instruct", type=str, default=DEFAULT_INSTRUCT)
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

def test_attack(attack, args):
	return attack.run(args.t, args.b, args.k, verbose=args.verbose)

def test_loss(attack: AdvAttack, args):
	func_loss = attack.get_target_ppl(attack.prompt).item()

	if args.verbose:
		print("Computed Loss", func_loss)

	logits = attack.model.forward(attack.prompt.unsqueeze(0)).logits
	logits = F.log_softmax(logits, dim = 2)
	target_logits = logits[0][attack.indices_dict["target"]-1]
	vals = attack.values_dict["target"]

	sum = 0
	for index in range(len(vals)):
		sum -= target_logits[index][vals[index]].item()

	if args.verbose:
		print("Computed Loss", sum)

	assert func_loss == sum
	
def test_candidates(attack: AdvAttack, args):
	candidates = attack.top_candidates(attack.prompt, attack.indices_dict["suffix"], attack.indices_dict["target"],100)

def test_prompt(attack: AdvAttack, args, suffix):
	attack.set_suffix(suffix)
	attack.prompt_response(verbose=args.verbose)

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

	for i in range(5):
		attack = AdvAttack(
			model, 
			tokenizer, 
			prompt=args.prompt, 
			target=args.target, 
			suffix_token = args.suffix_token, 
			suffix_length=args.suffix_length, 
			instruction=args.instruct
		)

		opt_suffix = test_attack(attack, args)
		print(opt_suffix)
		
		test_prompt(attack, args, opt_suffix)


	#test_loss(attack, args)
	#test_candidates(attack, args)
 
	# SUFFIX = torch.tensor([29442, 28017, 14161, 18164,  2033, 26682, 13531,  5384, 22308,  5384,
    #     10456, 20840,  7441, 15224,  8646, 23305, 23388, 18227,  2056, 25331,
    #     15236, 14099, 29833, 26909, 29962, 18294, 29588,  8643,  7521,  3621,
    #     30488,  5809,  1738,  6957, 16319, 21939, 28017,  1738,  5384,  1738,
    #     28017, 23192, 23388,  1738, 10834,  1738,  1738, 27466,  8703,  1738,
    #     10211,  1738,  1738,  1738,  1738,  1738,  1738,  1738,  1738, 15974,
    #       426, 14626,   426,   426]) ##gives RON WEASLEY



if __name__ == "__main__":
	main()


