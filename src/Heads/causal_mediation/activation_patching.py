from collections import defaultdict

import torch
from nnsight import LanguageModel
from torch import Tensor
from tqdm.auto import tqdm
from typing import Union
from Shared_utils.wrapper import ModelAccessor, get_accessor_config, get_model_specs


def activation_patching_batch(
    model: LanguageModel,
    clean_prompts: list[str]=None,
    corrupt_prompts: list[str]=None,
    answers: list[str]=None,
    batch_size: int = 8,
    remote: bool = True,
    pos: Union[list[int], str] = 'last_token',
) -> torch.Tensor:
    """
    Perform activation patching with batch processing.

    Args:
        model: Language model to analyze
        clean_prompts: List of original unmodified prompts
        corrupt_prompts: List of modified/corrupted prompts
        answers: List of expected answer strings
        batch_size: Size of each processing batch
        remote: Whether to run model remotely
        pos: position to patch: last_token, all_tokens, [](a list of positions)

    Returns:
        Tensor of intervention effects across all layers and heads
    """
    if pos == 'last_token':
        print("Patching last token")
    elif type(pos) == list:
        print(f"Patching tokens at positions: {pos}")
    elif pos == 'all_tokens':
        print("Patching all tokens")
    else:
        raise ValueError(f"Invalid position: {pos}")

    # # Input validation
    # total_samples = len(clean_prompts)
    # if not (total_samples == len(corrupt_prompts) == len(answers)):
    #     raise ValueError("All input lists must have the same length")

    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]

    # Tokenize all data upfront
    print("Tokenizing all data...")
    all_clean_tokens = model.tokenizer(
        clean_prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).input_ids

    if corrupt_prompts == None: 
        all_corrupt_tokens = all_clean_tokens
        mask = model.tokenizer("<|reserved_special_token_1|>", add_special_tokens=False).input_ids
        all_corrupt_tokens[:, 1:-2] = mask[0]
    else: 
        all_corrupt_tokens = model.tokenizer(
            corrupt_prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).input_ids
    token_len = all_corrupt_tokens.shape[1]
    # if pos != 'last_token':
    #     assert token_len == all_clean_tokens.shape[1]

    # Input validation
    total_samples = all_clean_tokens.shape[0]
    if not (all_clean_tokens.shape[0] == all_corrupt_tokens.shape[0]  == len(answers)):
        raise ValueError("All input lists must have the same length")

    # Get token IDs for answers
    answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

    # Initialize results storage
    if pos == 'all_tokens':
        all_results = torch.zeros((n_layers, token_len), device="cpu")
    else:
        all_results = torch.zeros((n_layers, n_heads), device="cpu")

    # Process in batches
    for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start

        # Get current batch tokens from pre-tokenized data
        clean_tokens = all_clean_tokens[batch_start:batch_end]
        corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
        batch_answer_tokens = answer_tokens[batch_start:batch_end]

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)

        # Step 1: Collect clean activations and logits
        clean_activations = {}
        clean_logits_values = []

        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(clean_tokens):
                for layer in range(n_layers):
                    if pos == 'last_token':
                        attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                        attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
                        for head in range(n_heads):
                            clean_activations[(layer, head)] = attn_reshaped[:, head, :].clone().save()
                    elif type(pos) == list:
                        attn_output = accessor.layers[layer].attention.output.unwrap().input[:, pos]
                        attn_reshaped = attn_output.reshape(current_batch_size, n_heads, len(pos), d_head)
                        for head in range(n_heads):
                            clean_activations[(layer, head)] = attn_reshaped[:, head, :].clone().save()
                    elif pos == 'all_tokens':
                        layer_output = accessor.layers[layer].unwrap().output[0] # [batch_size, seq_len, d_model]
                        for t in range(token_len):
                            clean_activations[(layer, t)] = layer_output[:, t, :].clone().save()
                    else:
                        raise ValueError(f"Invalid position: {pos}")

                # Get clean logits for answer tokens
                clean_logits = []
                for i in range(current_batch_size):
                    clean_logits.append(accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save())

        # Extract the actual tensor values after the trace is done
        if pos == 'all_tokens':
            for layer in range(n_layers):
                for t in range(token_len):
                    clean_activations[(layer, t)] = clean_activations[(layer, t)].value
        else:
            for layer in range(n_layers):
                for head in range(n_heads):
                    clean_activations[(layer, head)] = clean_activations[(layer, head)].value

        clean_logits_values = [logit.value for logit in clean_logits]

        # Step 2: Collect corrupt logits
        corrupt_logits_values = []

        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(corrupt_tokens):
                # Get corrupted logits for answer tokens
                corrupt_logits = []
                for i in range(current_batch_size):
                    corrupt_logits.append(accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save())

        corrupt_logits_values = [logit.value for logit in corrupt_logits]

        # Step 3: Run patching for each layer and head/pos
        if pos == 'all_tokens':
            batch_results = torch.zeros((n_layers, token_len), device="cpu")
        else:
            batch_results = torch.zeros((n_layers, n_heads), device="cpu")

        for layer in tqdm(
            range(n_layers),
            desc=f"Patching layers (batch {batch_start//batch_size + 1})",
            leave=False,
        ):
            if pos == 'all_tokens':
                for t in range(token_len):
                    patched_logits_values = []

                    with accessor.trace(remote=remote) as tracer:
                        with tracer.invoke(corrupt_tokens):
                            # Get corrupted layer output
                            
                            corrupt_layer_output = accessor.layers[layer].unwrap().output[0]

                            # patch the clean activation for this pos
                            corrupt_layer_output[:, t, :] = clean_activations[(layer, t)] # patch 

                            # Get patched logits
                            patched_logits = []
                            for i in range(current_batch_size):
                                patched_logits.append(
                                    accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save()
                                )
                    # Extract patched logits values
                    patched_logits_values = [logit.value for logit in patched_logits]

                    # Calculate intervention effects
                    effects = []
                    for i in range(current_batch_size):
                        clean_val = clean_logits_values[i]
                        corrupt_val = corrupt_logits_values[i]
                        patched_val = patched_logits_values[i]

                        # Avoid division by zero
                        denominator = clean_val - corrupt_val
                        if abs(denominator) < 1e-10:
                            effect = 0.0
                        else:
                            effect = (patched_val - corrupt_val) / denominator

                        effects.append(effect)

                    # Average effects across the batch
                    batch_results[layer, t] = sum(effects) / len(effects)

            else: 
                for head in range(n_heads):
                    patched_logits_values = []

                    with accessor.trace(remote=remote) as tracer:
                        with tracer.invoke(corrupt_tokens):
                            # Get corrupted attention output
                            if pos == 'last_token':
                                corrupt_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                                corrupt_reshaped = corrupt_attn_output.reshape(current_batch_size, n_heads, d_head)
                                
                            elif type(pos) == list:
                                corrupt_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, pos]
                                corrupt_reshaped = corrupt_attn_output.reshape(current_batch_size, n_heads, len(pos), d_head)
                                
                            else:
                                raise ValueError(f"Invalid position: {pos}")

                            # patch the clean activation for this head
                            corrupt_reshaped[:, head, :] = clean_activations[(layer, head)] # patch 

                            # Get patched logits
                            patched_logits = []
                            for i in range(current_batch_size):
                                patched_logits.append(
                                    accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save()
                                )

                    # Extract patched logits values
                    patched_logits_values = [logit.value for logit in patched_logits]

                    # Calculate intervention effects
                    effects = []
                    for i in range(current_batch_size):
                        clean_val = clean_logits_values[i]
                        corrupt_val = corrupt_logits_values[i]
                        patched_val = patched_logits_values[i]

                        # Avoid division by zero
                        denominator = clean_val - corrupt_val
                        if abs(denominator) < 1e-10:
                            effect = 0.0
                        else:
                            effect = (patched_val - corrupt_val) / denominator

                        effects.append(effect)

                    # Average effects across the batch
                    batch_results[layer, head] = sum(effects) / len(effects)

        # Accumulate results (weighted by batch size)
        all_results += batch_results * (current_batch_size / total_samples)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results
