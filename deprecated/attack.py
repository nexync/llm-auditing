import torch
import torch.nn.functional as F

import random
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import token_gradients

class AdvAttack():
	def __init__(self, model, tokenizer, prompt: str, target: str, suffix_token = "!", suffix_length = 16, instruction = ""):
		'''
			prompt[String]: Input question into model
			target[String]: Desired output from model
			instruction[String]: Instruction given to chat-tuned model
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
				indices_dict[keys[i]] = torch.tensor(range(running_index, running_index + len(chunk))).to(model.device)
				running_index += len(chunk)

			prompt = torch.cat(chunks, dim = 0).to(model.device)

			return prompt, indices_dict

		self.prompt, self.indices_dict = tokenize_inputs(prompt, target, suffix_token, suffix_length, instruction)
		self.target = self.tokenizer(target, return_tensors = "pt").input_ids[0][1:].to(model.device)
		
	def get_target_ppl(self, prompt):
		return -sum(torch.gather(F.log_softmax(self.model(prompt.unsqueeze(0)).logits, dim = 2)[0][self.indices_dict["target"]-1], 1, self.target.unsqueeze(1)))
	
	def change_suffix(self, token_id, index):
		'''
			index[int]: should be less than suffix_length
			token_id[int]: id of token to replace
		'''
		res = self.prompt.detach().clone()
		res[self.indices_dict["suffix"][0] + index] = token_id

		return res

	def top_candidates(self, input_tokens, gradient_indices, target_indices, k):
		grads = token_gradients(self.model, input_tokens, gradient_indices, target_indices) # T x V
		return grads.topk(k, dim = 1).indices # T x k
	
	def run(self, T, B, K, verbose = False):
		for i in tqdm.tqdm(range(1, T+1), disable = True):
			candidates = self.top_candidates(self.prompt, self.indices_dict["suffix"], self.indices_dict["target"], K)

			best_prompt_logprob = self.get_target_ppl(self.prompt)
			best_prompt = self.prompt
			for _ in range(B):
				r_index = random.randint(0, self.suffix_length-1)
				r_token = candidates[r_index][random.randint(0, K-1)]

				candidate_prompt = self.change_suffix(r_token, r_index)
				candidate_logprob = self.get_target_ppl(candidate_prompt)
				if best_prompt_logprob == None or candidate_logprob < best_prompt_logprob:
					best_prompt_logprob = candidate_logprob
					best_prompt = candidate_prompt
			
			self.prompt = best_prompt

			if verbose:
				print("iter ", i, "New suffix: ", self.tokenizer.decode(self.get_suffix()), " || ", "PPL: ", self.get_target_ppl(self.prompt).item())
			elif i % 100 == 0:
				print("iter ", i, "New suffix: ", self.tokenizer.decode(self.get_suffix()), " || ", "PPL: ", self.get_target_ppl(self.prompt).item())

		return self.get_suffix()

	def prompt_response(self, verbose = False):
		shortened_prompt = self.prompt[range(0, self.indices_dict["target"][0])]

		if verbose:
			print("Prompt: ", self.tokenizer.decode(shortened_prompt))

		output = self.model.generate(
			input_ids = shortened_prompt.unsqueeze(0), 
			attention_mask = torch.ones((1, shortened_prompt.shape[0])).to(self.model.device),
			max_new_tokens = 250,
		)

		print(self.tokenizer.decode(output[0]))

	def get_prompt(self):
		return self.prompt
	
	def set_suffix(self, new_suffix):
		assert new_suffix.shape[0] == self.suffix_length

		self.prompt[self.indices_dict["suffix"]] = new_suffix.to(self.model.device)

	def get_suffix(self):
		return self.prompt[self.indices_dict["suffix"]]
