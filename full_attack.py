import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import json
from attack import RandomGreedyAttack, CausalDPAttack, CausalDPAttackInitialized
import matplotlib.pyplot as plt

DEFAULT_INSTRUCT = "Answer all questions succinctly."

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--config_path", type=str, help="Optional config for attack parameters")

    parser.add_argument("-v", "--verbose", action="store_true", help="Show tqdm for runs")
    parser.add_argument("-a", "--attack_type", type=str, default="greedy", choices=["greedy", "causal"],
                        help="Type of attack to run")

    parser.add_argument("-q16", "--fp16", action="store_true", help="Use fp16 when loading in model")
    parser.add_argument("-q8", "--fp8", action="store_true", help="Load model with bitsandbytes 8bit")

    ######################################################
    #                CONFIG FILE PARAMS                  #
    ######################################################

    # Universal Params
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--instruct", type=str, default=DEFAULT_INSTRUCT)

    # Differs for greedy/causal
    parser.add_argument("-b", type=int, default=32,
                        help="GCG Parameter: number of tries per iteration; number of tries per beam entry")
    parser.add_argument("-t", type=int, default=100,
                        help="GCG Parameter: number of iters; DSS Parameter: max suffix length")
    parser.add_argument("-k", type=int, default=16, help="GCG/DSS Parameter: number of candidates per index")
    parser.add_argument("-e, --eval_log", type=bool, default=False,
                        help="GCG/DSS Parameter: whether to do greedy decode at each log step")

    # Greedy only params
    parser.add_argument("--suffix_token", type=str, default="!")
    parser.add_argument("--suffix_length", type=int, default=16)
    parser.add_argument("--log_freq", type=int, default=50, help=" GCG Parameter for logging")

    # Causal only params
    parser.add_argument("-m", type=int, default=8, help="DSS Parameter, beam width")

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


def prompt(attack, suffix=None):
    if suffix:
        attack.set_suffix(suffix)
    output = attack.greedy_decode_prompt()

    return output

def plot_ppls(ppls, output_file):
    x = list(range(1, len(ppls) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(x, ppls, marker='o', linestyle='-', color='b', label='Perplexity')
    plt.xlabel('qa pairs')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs QA pairs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)

def main():
    args = parse_args()

    if args.fp16:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
        if torch.cuda.is_available:
            model.to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        if torch.cuda.is_available:
            model.to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.verbose:
        print("Model and tokenizer loaded")

    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    with open(args.in_file, 'r') as file:
        qa_pairs = json.load(file)

    results = []
    ppls = []

    # Loop over each pair and perform attack
    if args.attack_type == "greedy":
        for qa in qa_pairs:
            a = RandomGreedyAttack(
                model,
                tokenizer,
                prompt=qa['question'],
                target=qa['response'],
                suffix_token=args.suffix_token,
                suffix_length=args.suffix_length,
                instruction=args.instruct
            )
            suffix, ppl = attack(a, args)
            ppls.append(ppl)
            output = prompt(a)
            # Store the result
            results.append({
                'question': qa['question'],
                'answer': qa['response'],
                'suffix': tokenizer.decode(suffix),
                'output': tokenizer.decode(output)
            })
            print(f'output: {tokenizer.decode(output)}')

    elif args.attack_type == "causal":
        for qa in qa_pairs:
            a = CausalDPAttack(
                model,
                tokenizer,
                prompt=qa['question'],
                target=qa['response'],
                instruction=args.instruct,
            )
            suffix = attack(a, args)
            # ppls.append(ppl)
            output = prompt(a)
            # Store the result
            results.append({
                'question': qa['question'],
                'answer': qa['response'],
                'suffix': tokenizer.decode(suffix),
                'output': tokenizer.decode(output)
            })
            print(f'output: {tokenizer.decode(output)}')
    else:
        raise Exception("Attack type unknown")

    # plot_ppls(ppls, 'ppls.png')
    with open(args.out_file, 'w') as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()


