from regex import D
import torch
import numpy as np
from nnsight import LanguageModel
from Shared_utils.wrapper import get_accessor_config, get_model_specs, ModelAccessor
from Shared_utils.shared_utils import *

def cache_correct_incorrect_head_output(
    model: LanguageModel, batch_size: int = 8,
    grouped_heads: dict[int, list[int]] = None, 
    remote: bool = False, 
    all_tokens_1: torch.Tensor = None,
    all_tokens_2: torch.Tensor = None,
    #all_answer_tokens: torch.Tensor = None,
) -> (dict, dict):
    """ 

    Args:
        model: Language model to analyze
        grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. 
        all_tokens_1: Pre-tokenized tokens 1
        all_tokens_2: Pre-tokenized tokens 2
        all_answer_tokens: Pre-tokenized answer tokens
        batch_size: Size of each processing batch
        remote: Whether to run model remotely
       
    Returns:
        all_act_1 (dict): activations of grouped heads,
             indexed by the tuple of each heads' layer and head index e.g. (12,5)
        all_act_2 (dict): activations of grouped heads, 
            indexed by the tuple of each heads' layer and head index e.g. (12,5)
    """
    

    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]

    # Input validation
    total_samples = all_tokens_1.shape[0]
    if not (total_samples == all_tokens_2.shape[0]):
        raise ValueError("All input lists must have the same length")

    # Initialize results storage 
    act_1 = {}
    act_2 = {}
    for layer in grouped_heads.keys():
        for head in grouped_heads[layer]:
            act_1[(layer, head)] = torch.empty(total_samples, d_head)
            act_2[(layer, head)] = torch.empty(total_samples, d_head)

    # Process in batches
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start

        # Get current batch tokens from pre-tokenized data
        batch_tokens_1 = all_tokens_1[batch_start:batch_end]
        batch_tokens_2 = all_tokens_2[batch_start:batch_end]
        #batch_answer_tokens = all_answer_tokens[batch_start:batch_end]

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)

        # Collect activations of grouped heads for tokens 1
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(batch_tokens_1):
                for layer in grouped_heads.keys(): # only need to cache relevant layers' activation 
                    attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
                    for head in grouped_heads[layer]: # only need to cache relevant heads' activation 
                        act_1[(layer, head)][batch_start:batch_end] = attn_reshaped[:, head, :].save()

        # Collect activations of grouped heads for tokens 2
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(batch_tokens_2):
                for layer in grouped_heads.keys(): # only need to cache relevant layers' activation 
                    attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
                    for head in grouped_heads[layer]: # only need to cache relevant heads' activation 
                        act_2[(layer, head)][batch_start:batch_end] = attn_reshaped[:, head, :].save()
    
    return act_1, act_2

def calculate_cosine_similarity_pairwise(
    tensor_1: np.ndarray, 
    tensor_2: np.ndarray
) -> np.ndarray:
    """
    Calculates the cosine similarity between each corresponding row (sample) 
    of two tensors.

    The input tensors must have the same shape (n_samples, d_head). 
    The output cosine_similarity is a 1D tensor of shape (n_samples), where the i-th element 
    is the cosine similarity between tensor_1[i, :] and tensor_2[i, :].

    Args:
        tensor_1: A numpy array of shape (n_samples, d_head).
        tensor_2: A numpy array of shape (n_samples, d_head).

    Returns:
        result (dict): cosine_similarity, norm_t1, norm_t2
    """
    tensor_1 = np.asarray(tensor_1)
    tensor_2 = np.asarray(tensor_2)

    # 1. Calculate the dot product (numerator) for each pair of rows.
    #    The operation (tensor_1 * tensor_2) is element-wise multiplication, 
    #    resulting in shape (n_samples, d_head). 
    #    Summing across axis=1 reduces the shape to (n_samples,).
    dot_product = np.sum(tensor_1 * tensor_2, axis=1)

    # 2. Calculate the L2 norm (magnitude) for each row of both tensors.
    #    np.linalg.norm with axis=1 also reduces the shape to (n_samples,).
    norm_t1 = np.linalg.norm(tensor_1, axis=1)
    norm_t2 = np.linalg.norm(tensor_2, axis=1)

    # 3. Calculate the product of the norms (denominator).
    denominator = norm_t1 * norm_t2

    # Handle cases where the norm is zero (a zero vector) to avoid division by zero.
    # If the denominator is close to zero, the similarity is set to 0.0 (or 1.0/0.0 
    # depending on whether the numerator is also zero, but 0.0 is safer).
    # np.divide performs element-wise division.
    similarities = np.divide(
        dot_product, 
        denominator, 
        out=np.zeros_like(dot_product, dtype=np.float32), 
        where=denominator != 0
    )
    result = {
        "cosine_similarity": similarities,
        'norm_t1': norm_t1,
        'norm_t2': norm_t2,
    }

    return result
