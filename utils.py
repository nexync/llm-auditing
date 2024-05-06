import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path, tokenizer_path = None):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16)
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        model.to("cuda:0")
        
    return model, tokenizer

def prompt_model(model, tokenizer, s, instruct = None):
    if instruct is None:
        instruct = "Answer all questions succinctly."
    full_string = " ".join(["[INST]", "<<SYS>>", instruct, "<</SYS>>", s, "[/INST]"])
    iids = tokenizer(full_string, return_tensors="pt").to(model.device)

    output = model.generate(**iids, max_new_tokens = 250)
    return tokenizer.decode(output[0])

def token_gradients(
    model,
    input_tokens,
    gradient_indices,
    target_indices,
):
    '''
        gradient_indices: assume that these are sorted, concurrent indices
    '''
    embedding_matrix = get_embedding_matrix(model)
    embeddings = get_embeddings(model, input_tokens.unsqueeze(0)).detach()

    one_hot_vec = F.one_hot(
        input_tokens[gradient_indices], 
        num_classes=embedding_matrix.shape[0],        
    ).half().to(model.device).requires_grad_(True)
    
    substitute_embeddings = (one_hot_vec @ embedding_matrix).unsqueeze(0)
    
    new_embeds = torch.cat([
        embeddings[:, :gradient_indices[0], :],
        substitute_embeddings,
        embeddings[:, gradient_indices[-1]+1:, :],
    ], dim = 1) # T x D

    logits = model(inputs_embeds=new_embeds).logits
    loss = nn.CrossEntropyLoss()(
        logits[0, target_indices, :],
        input_tokens[target_indices]
    )

    loss.backward()
    
    return one_hot_vec.grad.detach().clone()

def get_embedding_matrix(model):
    return model.model.embed_tokens.weight.detach().clone()

def get_embeddings(model, tokens):
    return model.model.embed_tokens(tokens)
