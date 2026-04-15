import numpy as np
import json
import os
import torch
from transformer_lens import HookedTransformer


def calculate_dynamic_score_abstract(
    model: HookedTransformer,
    prompts: list[str], 
    candidate_tokens: list[str],  
    k: int = 10
) -> tuple[float, np.ndarray]:
    """
    Calculate dynamic relation scores with per-head breakdown.
    For each head, calculates probability that at least one of the top-k tokens
    matches any token in the candidate_tokens list.
    
    Args:
        model: The language model
        prompts: List of input prompts
        candidate_tokens: List of tokens to look for (e.g., ['city', 'cities', 'capital', 'capitals'])
        k: Number of top tokens to consider
    
    Returns:
        - head_scores: (n_layers, n_heads) array of average scores per head
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_scores = np.zeros((n_layers, n_heads))
    
    # Convert candidate tokens to token IDs
    candidate_token_ids = set()
    for token in candidate_tokens:
        # Try both with and without leading space for robustness
        token_ids = model.tokenizer(token, add_special_tokens=False).input_ids
        candidate_token_ids.update(token_ids)
        
        # Also try with leading space (common tokenization pattern)
        spaced_token_ids = model.tokenizer(" " + token, add_special_tokens=False).input_ids
        candidate_token_ids.update(spaced_token_ids)
    
    # Process each prompt
    for prompt in prompts:
        prompt_tokens = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        _, cache = model.run_with_cache(
            prompt_tokens,
            names_filter=lambda name: "hook_z" in name
        )
        
        # Calculate scores for each layer and head
        for layer in range(n_layers):
            # Get attention output at last position
            z_output = cache[f"blocks.{layer}.attn.hook_z"][0, -1, :, :]
            
            # Project through W_O
            W_O = model.state_dict()[f"blocks.{layer}.attn.W_O"]
            head_outputs = torch.einsum('hd,hdo->ho', z_output, W_O)
            
            # Project to vocabulary
            W_U = model.state_dict()["unembed.W_U"]
            vocab_projections = head_outputs @ W_U
            
            for head in range(n_heads):
                _, top_k_tokens = torch.topk(vocab_projections[head], k)
                top_k_set = set(top_k_tokens.cpu().numpy())
                
                # Check if any top-k token matches any candidate token
                score = float(len(top_k_set.intersection(candidate_token_ids)) > 0)
                head_scores[layer, head] += score
    
    head_scores /= len(prompts)    
    return head_scores

def get_top_heads_from_score(
    scores:np.ndarray=None, n_top_heads:int=None
)->list[tuple[int, int, float]]:
    """
    Get a list of top heads from the score.
    """

    score_tensor = torch.from_numpy(scores).float()

    # Compute Top Heads (L,H)
    flattened_score = score_tensor.view(-1) # Flatten the tensor for topk (n_layers * n_heads)
    if n_top_heads is None:
        n_top_heads = flattened_score.shape[0]
    topk_vals, topk_inds = torch.topk(flattened_score, k=n_top_heads, largest=True) # Perform topk
    layer_indices, head_indices = torch.unravel_index(topk_inds, score_tensor.shape)
    top_lh_temp = torch.stack((layer_indices, head_indices), dim=1)

    # Combine the indices and values
    top_heads = []
    for i in range(n_top_heads):
        l_idx = top_lh_temp[i, 0].item()
        h_idx = top_lh_temp[i, 1].item()
        val = round(topk_vals[i].item(), 4)
        top_heads.append((l_idx, h_idx, val))
    
    return top_heads

def save_top_heads(prompt_type:str, exp_size:int, 
    top_heads:list[tuple[int, int, float]], 
    save_root:str, model_name:str, d_name:str, head_type:str
):
    """
    head_type: Relation_Propagation_heads 
    """
    components_dict = {'top_components':top_heads, 
            # 'grouped_top_components':grouped_heads_EP, 
    }

    ranked_components_file_path = os.path.join(save_root,
        model_name, d_name, "Heads", "MAPS", head_type, prompt_type,
    )

    if not os.path.exists(ranked_components_file_path):
        os.makedirs(ranked_components_file_path)

    with open(os.path.join(ranked_components_file_path,
        f"MAPS_ranked_top_heads_{exp_size}prompts.json"), 'w'
    ) as results_file:        
        json.dump(components_dict, results_file, indent=2)
    print(f"Saved {prompt_type} top heads to {ranked_components_file_path}")


def load_dataset(dataset_name, dataset_path):
    dataset_path = os.path.join(dataset_path, dataset_name, f"{dataset_name}.json")
    with open(dataset_path, "r") as f:
        dataset = json.loads(f.read())
    return dataset

def rearrange_heads_by_layer(heads):
    arranged = {}
    for (layer,head) in heads:
        arranged.setdefault(layer,[])
        arranged[layer].append(head)
    return arranged

def get_w_vo(layer, head, cfg, state_dict, is_gqa):
    if is_gqa:
      n_attn_groups = cfg.n_heads // cfg.n_key_value_heads
      value_head = head // n_attn_groups
      w_v = state_dict[f"blocks.{layer}.attn._W_V"][value_head]
    else:
      w_v = state_dict[f"blocks.{layer}.attn.W_V"][head]
    w_o = state_dict[f"blocks.{layer}.attn.W_O"][head]
    w_vo = w_v @ w_o
    return w_vo

def top_k_indices(matrix, k):
    sorted_indices = np.argsort(matrix.flatten(), kind='stable')[::-1][:k]
    return np.array(np.unravel_index(sorted_indices, matrix.shape)).T

def top_k_indices_old(matrix, k):
    flat_indices = np.argpartition(matrix.ravel(), -k)[-k:]
    # Sort the top-k indices by value (descending order)
    top_k_sorted = flat_indices[np.argsort(matrix.ravel()[flat_indices], kind='stable')[::-1]]
    # Convert flat indices back to (i, j) indices
    return np.array(np.unravel_index(top_k_sorted, matrix.shape)).T

# def get_k(model_name, relation_name):
#     if "Llama-3" in model_name:
#         k = 3 if "copying" in relation_name else 10 #TODO: adjust this 
#     else:
#         k = 1 if "copying" in relation_name else 10
#     return k

def get_topm_relation_heads(m, maps, dataset, apply_first_mlp, k, only_nonzero):
    relation_scores, _ = maps.calc_relation_scores(dataset, apply_first_mlp, k)
    top_k_heads = top_k_indices(relation_scores, m)
    if only_nonzero:
        top_k_heads = [(layer,head) for (layer,head) in top_k_heads if relation_scores[layer,head] > 0.0]
    return top_k_heads
