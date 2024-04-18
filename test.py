import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
from attack import AdvAttack

DEFAULT_PROMPT = "How many children do Arthur and Molly Weasley have? " #In what fictional bank did Harry Potter keep his money?" #"Who are Harry Potter's best friends?"
DEFAULT_TARGET = "Arthur and Molly Weasley have seven children." #"Gringotts" #"Harry Potter's best friends are Ron Weasley and Hermione Granger."
DEFAULT_INSTRUCT = "Answer all questions succinctly."

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True, help="Path to model")
	parser.add_argument("-q", "--fp16", action="store_true", help="Use fp16 when loading in model")
	parser.add_argument("-v", "--verbose", action="store_true", help="Show tqdm for runs")

	args = parser.parse_args()

	return args

def test_attack(attack, args):
	attack.run(256, 64, 128, verbose=args.verbose)

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
	return

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

	attack = AdvAttack(
		model, 
		tokenizer, 
		prompt=DEFAULT_PROMPT, 
		target=DEFAULT_TARGET, 
		suffix_token = "!", 
		suffix_length=64, 
		instruction=DEFAULT_INSTRUCT
	)

	test_attack(attack, args)
	#test_loss(attack, args)
	#test_candidates(attack, args)
 
	# SUFFIX = torch.tensor([ 1738,  1738,  1738, 11958,  1738,    40,    86,  6745,  1738,  1738,
    #      1738, 15224, 31296,   195, 11287, 19758,  1738, 25558,  3441, 27818,
    #     20833, 30226, 30140, 15608, 29646, 16668,  1738,  1738,  1738, 16924,
    #      1738, 18227, 29387,   123,   255,  1841,    79,    79,  6780,  1738,
    #      2880, 14120, 15211, 18306,  4156, 21546, 30210,    57,    55,   111,
    #     11864,  5754, 30121,  1738, 14777, 17686,  5919, 20273,   890, 16826,
    #      1738,  1738, 15639,  8585])
	# SUFFIX = torch.tensor([ 1738, 13401, 25632,    35,  1738,  1738, 10382,  1738, 25257, 22633,
    #      6121,   123, 23716,  3277,  1738, 26054,  1738,  6710,  8598,    46,
    #     27189, 21447, 14861,  1738,  8981,   984, 14700,  1738,  1738, 29534,
    #     26139,  1738, 14833,  1738, 21790,  1738, 25064,  1738, 26736,  1738,
    #      1738, 13955, 29883,  1738, 27692, 25747,    87, 19937, 15286,  1738,
    #      1410,    50,  3180,  2503,  1738,  1738, 12305,  1213,  4852,  9156,
    #      5124, 25553,  1738, 15063])
	
	# test_prompt(attack, args, SUFFIX)

if __name__ == "__main__":
	main()


