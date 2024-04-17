import torch
import torch.nn.functional as F

import random
import tqdm

from utils import token_gradients

class AdvAttack():
	def __init__(self, model, tokenizer, prompt: str, target: str, suffix_token = "!", suffix_length = 16, instruction = ""):
		'''
			prompt[String]: Input question into model
			target[String]: Desired output from model
			suffix_token[Char]: Default suffix token when initializing attack
			suffix_length[Int]: Allowed number of tokens in adversarial suffix
		'''
		self.model = model
		self.tokenizer = tokenizer
		self.suffix_length = suffix_length

		def tokenize_inputs(prompt, target, suffix_token, suffix_length, instruction):
			chunks = [
				self.tokenizer("<s>", return_tensors = "pt").input_ids[0][1:],
				self.tokenizer("[INST]", return_tensors = "pt").input_ids[0][1:],
				self.tokenizer(" ".join(["<<SYS>>", instruction, "<</SYS>>"]), return_tensors = "pt").input_ids[0][1:],
				self.tokenizer(prompt, return_tensors = "pt").input_ids[0][1:],
				torch.tensor([self.tokenizer(suffix_token).input_ids[1]]*suffix_length),
				self.tokenizer("[/INST]", return_tensors = "pt").input_ids[0][1:],
				self.tokenizer(target, return_tensors = "pt").input_ids[0][1:],
				self.tokenizer("</s>", return_tensors = "pt").input_ids[0][1:],
			]

			running_index = 0
			indices_dict = {}
			values_dict = {}

			keys = {
				0: "bos",
				1: "boi",
				2: "instruct",
				3: "prompt",
				4: "suffix",
				5: "eoi",
				6: "target",
				7: "eos",
			}

			for i, chunk in enumerate(chunks):
				indices_dict[keys[i]] = list(range(running_index, running_index + len(chunk)))
				values_dict[keys[i]] = chunk.to(model.device)
				running_index += len(chunk)

			prompt = torch.cat(chunks, dim = 0).to(model.device)

			return prompt, indices_dict, values_dict

		self.prompt, self.indices_dict, self.values_dict = tokenize_inputs(prompt, target, suffix_token, suffix_length, instruction)
		
	def get_target_ppl(self, prompt):
		return -sum(torch.gather(F.log_softmax(self.model(prompt.unsqueeze(0)).logits, dim = 2)[0][self.indices_dict["target"]-1], 1, self.values_dict["target"].unsqueeze(1)))
	
	def update_suffix(self, token_id, index):
		'''
			index[int]: should be less than suffix_length
			token_id[int]: id of token to replace
		'''
		res = self.prompt.detach().clone()
		res[self.indices_dict["suffix"][0] + index] = token_id

		return res

	def top_candidates(self, input_tokens, gradient_indices, target_indices, k):
		grads = token_gradients(self.model, input_tokens, gradient_indices, target_indices)
		return grads.topk(k, dim = 0).indices
	
	def run(self, T, B, k, verbose = False):
		for _ in tqdm.tqdm(range(T), disable = not verbose):
			candidates = self.top_candidates(self.prompt, self.indices_dict["suffix"], self.indices_dict["targets"], k)

			best_prompt_loss = self.get_target_ppl(self.prompt)
			best_prompt = self.prompt
			for _ in range(B):
				r_index = random.randint(0, self.suffix_length-1)
				r_token = candidates[r_index][random.randint(0, k)]

				candidate_prompt = self.update_suffix(r_token, r_index)
				candidate_ppl = self.get_target_ppl(candidate_prompt)
				if best_prompt_loss == None or candidate_ppl < best_prompt_loss:
					best_prompt_loss = candidate_ppl
					best_prompt = candidate_prompt
			
			self.prompt = best_prompt

			if verbose:
				print("New suffix: ", self.get_suffix(), " || ", "PPL: ", self.get_target_ppl())

	def get_prompt(self):
		return self.prompt

	def get_suffix(self):
		return self.prompt[self.indices_dict["suffix"]]