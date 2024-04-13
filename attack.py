import torch

from utils import token_gradients

class AdvAttack():
	def __init__(self, model, tokenizer, prompt: str, target: str, suffix_token = "!", suffix_length = 16):
		'''
			prompt[String]: Input question into model
			target[String]: Desired output from model
			suffix_token[Char]: Default suffix token when initializing attack
			suffix_length[Int]: Allowed number of tokens in adversarial suffix
		'''
		self.model = model
		self.tokenizer = tokenizer

		def tokenize_inputs(prompt, target, suffix_token, suffix_length):
			return_dict = {
				"prompt_tokens": self.tokenizer(prompt, return_tensors = "pt").input_ids[0][1:],
				"target_tokens": self.tokenizer(target, return_tensors = "pt").input_ids[0][1:],
				"suffix_tokens": torch.tensor([self.tokenizer(suffix_token).input_ids[1]]*suffix_length),
			}
			length_dict = {
				"prompt": len(return_dict["prompt_tokens"]),
				"target": len(return_dict["target_tokens"]),
				"suffix": len(return_dict["suffix_tokens"]),
			}

			return return_dict, length_dict
		
		def tokenize_special_chars():
			return_dict = {
				"start": self.tokenizer("<s>").input_ids[1:],
				"end": self.tokenizer("</s>").input_ids[1:],
				"inst_start": self.tokenizer("[INST]").input_ids[1:],
				"inst_end": self.tokenizer("[/INST]").input_ids[1:],
			}

		self.token_ids, self.lengths = tokenize_inputs(prompt, target, suffix_token, suffix_length)

	def top_candidates(
		self,
		model,
		input_tokens,
		gradient_indices,
		target_indices,
		k
	):
		grads = token_gradients(model, input_tokens, gradient_indices, target_indices)
		return grads.topk(k, dim = 0)

	def format_qa_pair(self, question, suffix, answer):
		return " ".join(["[INST]", question, suffix, "[/INST]", answer, "</s>"])

