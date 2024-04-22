import numpy as np
import torch
import torch.nn.functional as F

import random
import tqdm
import time 

from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.profiler import profile, record_function, ProfilerActivity

from utils import token_gradients

random.seed(42)

class BaseAdvAttack():
	def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, query: str, target: str, max_suffix_length = 64, instruction = ""):
		'''
			query[String]: Input question into model
			target[String]: Desired output from model
			instruction[String]: Instruction given to chat-tuned model
		'''
		self.model = model
		self.tokenizer = tokenizer
		self.max_suffix = max_suffix_length

		self.target = self.tokenizer(target, return_tensors = "pt").input_ids[0][1:].to(model.device)
		self.eoi = self.tokenizer("[/INST]", return_tensors = "pt").input_ids[0][1:].to(model.device)

		def tokenize_inputs(query, target, instruction):
			pre_suffix_chunk = [
				self.tokenizer("<s>", return_tensors = "pt").input_ids[0][1:],
				self.tokenizer("[INST]", return_tensors = "pt").input_ids[0][1:],
				self.tokenizer(" ".join(["<<SYS>>", instruction, "<</SYS>>"]), return_tensors = "pt").input_ids[0][1:],
				self.tokenizer(query, return_tensors = "pt").input_ids[0][1:],
			]

			post_suffix_chunk = [
				self.eoi,
				self.target,
				self.tokenizer("</s>", return_tensors = "pt").input_ids[0][1:].to(model.device),
			]

			running_index = 0
			indices_dict = {}

			keys = {
				0: "bos",
				1: "boi",
				2: "instruct",
				3: "query",
				4: "eoi",
				5: "target",
				6: "eos",
			}

			for i, chunk in enumerate(pre_suffix_chunk + post_suffix_chunk):
				indices_dict[keys[i]] = torch.tensor(range(running_index, running_index + len(chunk))).to(model.device)
				running_index += len(chunk)

			pre_suffix = torch.cat(pre_suffix_chunk, dim = 0).to(model.device)
			post_suffix = torch.cat(post_suffix_chunk, dim = 0).to(model.device)

			return pre_suffix, post_suffix, indices_dict
		
		self.pre_suffix, self.post_suffix, self.indices_dict = tokenize_inputs(query, target, instruction)

		self.suffix_start = self.pre_suffix.shape[0]
		self.suffix = torch.tensor([]).to(model.device)

	def get_input(self, alternate_suffix = None):
		'''Returns entire token id sequence on which optimization is performed. Used during optimization.'''
		if alternate_suffix is not None:
			return torch.cat([self.pre_suffix, alternate_suffix, self.post_suffix], dim = 0) # L
		else:
			return torch.cat([self.pre_suffix, self.suffix, self.post_suffix], dim = 0) # L
	
	def get_prompt(self, alternate_suffix = None):
		'''Returns token id sequence that will be inputted into the model to try to greedy decode target sequence.'''
		if alternate_suffix is not None:
			return torch.cat([self.pre_suffix, alternate_suffix, self.eoi], dim = 0)
		else:		
			return torch.cat([self.pre_suffix, self.suffix, self.eoi], dim = 0)
		
	def get_suffix_indices(self):
		return torch.tensor(range(self.suffix_start, self.suffix_start + self.suffix.shape[0]), device = self.model.device)
	
	def get_target_suprisal_unbatched(self, input, target_indices, reduction = "sum"):
		with torch.no_grad():
			logprobs = (1 / np.log(2.)) * F.log_softmax(self.model(input).logits, dim = 2)[0] # B x L x V
			logprobs = logprobs[target_indices] # B x S x V
			logprobs = torch.gather(logprobs, 1, self.target.unsqueeze(1)) # B x S x 1
			loss = -logprobs.sum()

		if reduction == "sum":
			return loss
		elif reduction == "mean":
			return (loss / self.target.shape(0))
		else:
			raise Exception("Unknown reduction type")

	def get_target_surprisal(self, input, target_indices, reduction = "sum"):
		'''
			input: B x L
		'''
		b, _ = input.shape
		with torch.no_grad():
			logprobs = (1 / np.log(2.)) * F.log_softmax(self.model(input).logits, dim = 2) # B x L x V
			logprobs = logprobs[:, target_indices] # B x S x V
			logprobs = torch.gather(logprobs, 2, self.target.unsqueeze(1).repeat(b, 1, 1)) # B x S x 1
			loss = -logprobs.sum(dim = 1).squeeze(1) # B

		if reduction == "sum":
			return loss
		elif reduction == "mean":
			return (loss / self.target.shape(0))
		else:
			raise Exception("Unknown reduction type")
	
	def get_target_ppl(self, input, reduction = "sum"):
		surprisal = self.get_target_surprisal(input, reduction=reduction)
		return 2**surprisal
	
	def top_candidates(self, input_tokens, gradient_indices, target_indices, k):
		grads = token_gradients(self.model, input_tokens, gradient_indices, target_indices) # T x V
		return grads.topk(k, dim = 1).indices # T x k
	
	def set_suffix(self, suffix):
		self.suffix = suffix
	
	def update_suffix(self, token_id, index):
		'''Function used to generate a new suffix. Index of -1 implies adding to suffix length'''
		res = self.suffix.detach().clone()
		assert index == -1 or 0 <= index < res.shape[0], "Invalid index to update"

		if index == -1:
			res = torch.cat([res, torch.tensor([token_id], dtype=res.dtype, device=res.device)])
		else:
			res[index] = token_id

		return res
	
	def greedy_decode_prompt(self, alternate_prompt = None, max_new_tokens = 512):
		if alternate_prompt:
			prompt = alternate_prompt
		else:
			prompt = self.get_prompt()

		with torch.no_grad():
			output = self.model.generate(
				input_ids = prompt.unsqueeze(0), 
				attention_mask = torch.ones((1, prompt.shape[0])).to(self.model.device),
				max_new_tokens = max_new_tokens,
			)
		return output[0]
	
	def run(self, **params):
		'''Implemented by derived classes'''
		raise NotImplementedError


class RandomGreedyAttack(BaseAdvAttack):
	def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, target: str, suffix_token = "!", suffix_length = 64, instruction = ""):
		super().__init__(model, tokenizer, prompt, target, max_suffix_length=suffix_length, instruction=instruction)
		self.suffix = torch.tensor([self.tokenizer(suffix_token).input_ids[1]]*suffix_length, device = model.device)

	def run(self, **params):
		'''
			params:
				- T[int]: number of iterations attack is run
				- B[int]: number of substitutions attempted per iteration
				- K[int]: number of candidates per gradient index
				- batch_size[int]: number of items per batch
				- log_freq[int]: how often to log intermediate steps
				- eval_log[bool]: whether to run prompt eval during logging
		'''
		defaults = {"log_freq": 10, "eval_log": False, "verbose": False, "batch_size": 16}
		params = {**defaults, **params}
		assert min([key in params for key in ["T", "B", "K"]]), "Missing arguments in attack"

		for iter in tqdm.tqdm(range(1, params["T"]+1), initial=1):
			curr_input = self.get_input()
			candidates = self.top_candidates(
				curr_input, 
				self.get_suffix_indices(),
				self.indices_dict["target"] + self.suffix.shape[0],
				params["K"]
			)
			
			best_surprisal = self.get_target_surprisal(
				curr_input.unsqueeze(0),
				self.indices_dict["target"] + self.suffix.shape[0] - 1,
			)[0].item()
			best_suffix = self.suffix

			input_batch = []
			suffix_batch = []

			for index in range(params["B"]):
				r_index = random.randint(0, self.suffix.shape[0]-1)
				r_token = candidates[r_index][random.randint(0, params["K"]-1)]

				candidate_suffix = self.update_suffix(r_token, r_index)
				candidate_input = self.get_input(alternate_suffix=candidate_suffix)

				suffix_batch.append(candidate_suffix)
				input_batch.append(candidate_input)

				if len(input_batch) == params["batch_size"] or index == params["B"] - 1:
					if params["batch_size"] == 1:
						candidate_surprisals = self.get_target_suprisal_unbatched(
							input_batch[0].unsqueeze(0),
							self.indices_dict["target"] + candidate_suffix.shape[0] - 1,
						)
						batch_best = candidate_surprisals

					else:
						candidate_surprisals = self.get_target_surprisal(
							torch.stack(input_batch, dim = 0),
							self.indices_dict["target"] + candidate_suffix.shape[0] - 1,
						) # B

						batch_best = torch.min(candidate_surprisals)

					if batch_best < best_surprisal:
						best_surprisal = batch_best
						best_suffix = suffix_batch[torch.argmin(candidate_surprisals)]

					del candidate_surprisals
					suffix_batch = []
					input_batch = []
									
			self.suffix = best_suffix

			if iter % params["log_freq"] == 0:
				print("iter ", iter, " || ", "PPL: ", best_surprisal.item())

				if params["eval_log"]:
					print("Suffix: ", self.tokenizer.decode(best_suffix))

					if params["verbose"]:
						print("Output: ", self.tokenizer.decode(self.greedy_decode_prompt()))
			
		return self.suffix
		
class CausalGreedyAttack(BaseAdvAttack):
	def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, query: str, target: str, max_suffix_length=64, instruction=""):
		super().__init__(model, tokenizer, query, target, max_suffix_length, instruction)

	def run(self, **params):
		'''
			params:
				- T[int]: number of iterations attack is run
				- B[int]: number of substitutions attempted per iteration
				- K[int]: number of candidates per gradient index
				- log_freq[int]: how often to log intermediate steps
				- eval_log[bool]: whether to run prompt eval during logging
		'''
		defaults = {"log_freq": 10, "eval_log": False}
		params = {**defaults, **params}
		assert min([key in params for key in ["T", "B", "K"]]), "Missing arguments in attack"