import torch
import torch.nn as nn
import torch.nn.functional as F

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
    embeddings = get_embeddings(input_tokens)

    one_hot_vec = F.one_hot(
        input_tokens[gradient_indices], 
        num_classes=embedding_matrix.shape[0],        
    ).float().to(model.device).requires_grad_(True)
    
    substitute_embeddings = (one_hot_vec @ embedding_matrix).unsqueeze(0)
    new_embeds = torch.cat([
        embeddings[:, :gradient_indices[0], :],
        substitute_embeddings,
        embeddings[:, gradient_indices[-1]:, :],
    ], dim = 1)

    logits = model(input_embeds=new_embeds).logits
    loss = nn.CrossEntropyLoss()(
        logits[0, target_indices, :],
        input_tokens[target_indices]
    )

    loss.backward()
    
    return one_hot_vec.grad.clone()


def get_embedding_matrix(model):
    return model.model.embed_tokens.weight.detatch().clone()

def get_embeddings(model, tokens):
    return model.model.embed_tokens(tokens)