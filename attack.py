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
			chunks = [
				self.tokenizer("<s>", return_tensors = "pt").input_ids[0][1:],
				self.tokenizer("[INST]", return_tensors = "pt").input_ids[0][1:],
				self.tokenizer(prompt, return_tensors = "pt").input_ids[0][1:],
				torch.tensor([self.tokenizer(suffix_token).input_ids[1]]*suffix_length),
				self.tokenizer("[/INST]", return_tensors = "pt").input_ids[0][1:],
				self.tokenizer(target, return_tensors = "pt").input_ids[0][1:],
				self.tokenizer("</s>", return_tensors = "pt").input_ids[0][1:],
			]

			running_index = 0
			length_dict = {}
			id_dict = {}
			for i, chunk in enumerate(chunks):
				length_dict[i] = list(range(running_index, running_index + len(chunk)))
				id_dict[i] = chunk
				running_index += len(chunk)


			prompt = torch.cat(chunks, dim = 0)

			return prompt, length_dict, id_dict

		self.prompt, self.length_dict, self.id_dict = tokenize_inputs(prompt, target, suffix_token, suffix_length)
		
	def get_target_ppl(self):
		return sum(torch.gather(self.model(self.prompt.unsqueeze(0)).logits[0][self.length_dict[5]], 1, self.id_dict[5].unsqueeze(1)))
	
	def update_suffix(self, token_id, index):
		'''
			index[int]: should be less than suffix_length
			token_id[int]: id of token to replace
		'''
		self.prompt[self.length_dict[3][0] + index] = token_id

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

