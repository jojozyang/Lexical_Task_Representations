import torch
import os
import json
from nnsight import LanguageModel

from Shared_utils.shared_utils import (group_ranked_locations_by_layer, compute_token_rank_prob,
    compute_top_k_accuracy, get_head_activation_ranks_probs_mean, sample_random_heads, free_unused_cuda_memory)
from Shared_utils.wrapper import get_accessor_config, get_model_specs, ModelAccessor
from Shared_utils.prompt_utils import generate_few_shot_prompts, generate_instruction_prompts
from Heads.Function_vectors.utils.utils import get_vector_from_universal_top_indirect_effect_heads

from typing import Union

def set_activation_patching_batch(
    model: LanguageModel,
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    answers: list[str],
    patch_positions: list[tuple[int, int]],  # List of (layer, head) pairs to patch
    batch_size: int = 8,
    prepend_bos: bool = True,
    remote: bool = False,
    return_effect: bool = True,  # If True, return intervention effect; if False, return raw logits
) -> torch.Tensor:
    """
    Perform set activation patching - patch multiple (layer, head) pairs simultaneously.

    Args:
        model: Language model to analyze
        clean_prompts: List of original unmodified prompts
        corrupt_prompts: List of modified/corrupted prompts
        answers: List of expected answer strings
        patch_positions: List of (layer, head) tuples to patch simultaneously
        batch_size: Size of each processing batch
        remote: Whether to run model remotely
        return_effect: If True, return intervention effect; if False, return raw patched logits

    Returns:
        Tensor of average results across all samples
    """
    # Input validation
    total_samples = len(clean_prompts)
    if not (total_samples == len(corrupt_prompts) == len(answers)):
        raise ValueError("All input lists must have the same length")

    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]

    # Tokenize all data upfront
    #print("Tokenizing all data...")
    all_clean_tokens = model.tokenizer(
        clean_prompts,
        padding=True,
        padding_side="left",
        add_special_tokens=prepend_bos,
        return_tensors="pt",
    ).input_ids

    all_corrupt_tokens = model.tokenizer(
        corrupt_prompts,
        padding=True,
        padding_side="left",
        add_special_tokens=prepend_bos,
        return_tensors="pt",
    ).input_ids

    # Get token IDs for answers
    answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

    # Initialize results storage
    all_results = []

    # Process in batches
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start

        # Get current batch tokens from pre-tokenized data
        clean_tokens = all_clean_tokens[batch_start:batch_end]
        corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
        batch_answer_tokens = answer_tokens[batch_start:batch_end]

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)

        # Step 1: Collect clean activations for all patch positions
        clean_activations = {}

        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(clean_tokens):
                # Collect activations for all layers that we need to patch
                layers_to_collect = set(layer for layer, _ in patch_positions)
                
                for layer in layers_to_collect:
                    attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
                    
                    # Store activations for all heads in this layer that we need
                    heads_in_layer = [head for l, head in patch_positions if l == layer]
                    for head in heads_in_layer:
                        clean_activations[(layer, head)] = attn_reshaped[:, head, :].clone().save()

                # Get clean logits for answer tokens (if we need them for effect calculation)
                if return_effect:
                    clean_logits = []
                    for i in range(current_batch_size):
                        clean_logits.append(accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save())

        # Extract the actual tensor values after the trace is done
        for layer, head in patch_positions:
            clean_activations[(layer, head)] = clean_activations[(layer, head)].value

        if return_effect:
            clean_logits_values = [logit.value for logit in clean_logits]

        # Step 2: Get corrupt logits (if needed for effect calculation)
        if return_effect:
            corrupt_logits_values = []
            with accessor.trace(remote=remote) as tracer:
                with tracer.invoke(corrupt_tokens):
                    corrupt_logits = []
                    for i in range(current_batch_size):
                        corrupt_logits.append(accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save())
            corrupt_logits_values = [logit.value for logit in corrupt_logits]

        # Step 3: Run patching for ALL specified positions simultaneously
        patched_logits_values = []

        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(corrupt_tokens):
                # Patch ALL specified positions in this single forward pass
                for layer, head in patch_positions:
                    # Get corrupted attention output for this layer
                    corrupt_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    corrupt_reshaped = corrupt_attn_output.reshape(current_batch_size, n_heads, d_head)
                    
                    # Patch in the clean activation for this head
                    corrupt_reshaped[:, head, :] = clean_activations[(layer, head)]

                # Get patched logits after all patches are applied
                patched_logits = []
                for i in range(current_batch_size):
                    patched_logits.append(
                        accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save()
                    )

        # Extract patched logits values
        patched_logits_values = [logit.value for logit in patched_logits]

        # Calculate results for this batch
        if return_effect:
            # Calculate intervention effects
            batch_effects = []
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

                batch_effects.append(effect)
            
            all_results.extend(batch_effects)
        else:
            # Just return the raw patched logits
            all_results.extend(patched_logits_values)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Return average across all samples
    return sum(all_results) / len(all_results)


def eval_head_batch_pair_wise(
    model: LanguageModel,
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    answers: list[str],
    batch_size: int = 8,
    remote: bool = False,
    grouped_heads: dict[int, list[int]] = None, 
    scaling_factor: float = 1.0,
    patch_pos: Union[int, str] = -1,
    prepend_bos:int = True,
    intervention_type: str = "ablation",
) -> torch.Tensor:
    """
    TODO: the code for getting logits might be sepcific to llama model for now 
    Evaluate the effect of activating or ablating some heads by patching pair-wise activations. 
    Base run: collect the activation to be patched 
    Intervention run: collect logits after intervention 

    Args:
        model: Language model to analyze
        clean_prompts: List of prompts
        corrupt_prompts: List of prompts
        answers: List of expected answer strings
        batch_size: Size of each processing batch
        remote: Whether to run model remotely
        grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. Useful for interventions
        scaling_factor: scaling factor for the corruption head activation
        patch_pos: position to patch the corruption head activation. 
            -1: patch the last position
            "all": patch all positions
        intervention_type: ablation or activation 

    Returns:
        base_ranks (torch.Tensor):
        base_probs (torch.Tensor):
        intervention_ranks (torch.Tensor):
        intervention_probs (torch.Tensor):
    """
    # Input validation
    total_samples = len(clean_prompts)
    if not (total_samples == len(corrupt_prompts) == len(answers)):
        raise ValueError("All input lists must have the same length")

    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
    d_head = d_model // n_heads

    # Tokenize all data upfront
    #print("Tokenizing all data...")
    all_clean_tokens = model.tokenizer(
        clean_prompts,
        padding=True,
        padding_side="left",
        add_special_tokens=prepend_bos,
        return_tensors="pt",
    ).input_ids

    all_corrupt_tokens = model.tokenizer(
        corrupt_prompts,
        padding=True,
        padding_side="left",
        add_special_tokens=prepend_bos,
        return_tensors="pt",
    ).input_ids

    # Get token IDs for answers
    answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

    # Initialize results storage # TODO 
    base_ranks = []
    base_probs = []
    intervention_ranks = []
    intervention_probs = []

    # Process in batches
    # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start

        # Get current batch tokens from pre-tokenized data
        clean_tokens = all_clean_tokens[batch_start:batch_end]
        corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
        batch_answer_tokens = answer_tokens[batch_start:batch_end]
        if intervention_type == "ablation":
            base_tokens = corrupt_tokens
            interv_tokens = clean_tokens
        elif intervention_type == "activation":
            base_tokens = clean_tokens
            interv_tokens = corrupt_tokens

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)

        # Step 1: Collect base activations and logits
        base_activations = {}

        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(base_tokens):
                for layer in grouped_heads.keys(): # only need to cache relevant layers' activation 
                    attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)

                    for head in grouped_heads[layer]: # only need to cache relevant heads' activation 
                        base_activations[(layer, head)] = attn_reshaped[:, head, :].clone().save()

                base_logits = model.output.logits[:,-1].save()
                # Get clean logits for answer tokens
                # clean_logits = []
                # for i in range(current_batch_size):
                #     clean_logits.append(accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save())

        # Run patching 
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(interv_tokens):
                for layer in grouped_heads.keys():
                    if patch_pos == -1:
                        interv_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                        interv_reshaped = interv_attn_output.reshape(current_batch_size, n_heads, d_head)
                        for head in grouped_heads[layer]: 
                            # patch activation for certain heads in this layer 
                            interv_reshaped[:, head, :] = scaling_factor * base_activations[(layer, head)] 

                    elif patch_pos == "all":
                        # base_attn_output = accessor.layers[layer].attention.output.unwrap().input
                        # base_reshaped = base_attn_output.reshape(current_batch_size, clean_token_len, n_heads, d_head)

                        # # patch the activation for certain heads in this layer 
                        # base_reshaped[:, :, grouped_heads[layer], :] = scaling_factor * patching_activations[layer, grouped_heads[layer]] 
                        raise ValueError("patch_pos = all not implemented")

                    else:
                        raise ValueError(f"Invalid patch position: {patch_pos}")

                interv_logits = model.output.logits[:,-1].save() #TODO: confirm this is right 
                # Get patched logits
                # patched_logits = []
                # for i in range(current_batch_size):
                #     patched_logits.append(
                #         accessor.lm_head.unwrap().output[i, -1, batch_answer_tokens[i]].save()
                #     )

        batch_base_ranks, batch_base_probs = compute_token_rank_prob(base_logits, batch_answer_tokens)
        batch_intervention_ranks, batch_intervention_probs = compute_token_rank_prob(interv_logits, batch_answer_tokens)    
        del interv_logits, base_logits, base_activations # free memory
        free_unused_cuda_memory()       

        base_ranks.append(batch_base_ranks)
        base_probs.append(batch_base_probs)
        intervention_ranks.append(batch_intervention_ranks)
        intervention_probs.append(batch_intervention_probs)
    
    base_ranks = torch.cat(base_ranks)
    base_probs = torch.cat(base_probs)
    intervention_ranks = torch.cat(intervention_ranks)
    intervention_probs = torch.cat(intervention_probs)

    return base_ranks, base_probs, intervention_ranks, intervention_probs

def eval_head_batch(
    model: LanguageModel,
    base_prompts: list[str],
    answers: list[str],
    batch_size: int = 8,
    remote: bool = False,
    patching_activations: torch.Tensor = None, 
    grouped_heads: dict[int, list[int]] = None, 
    scaling_factor: float = 1.0,
    patch_pos: Union[int, str] = -1,
) -> torch.Tensor:
    """
    Evaluate the effect of ablating or activating some heads by patching the corrupt or clean mean activations. 

    Args:
        model: Language model to analyze
        base_prompts: List of prompts
        corrupt_prompts: List of prompts
        answers: List of expected answer strings
        batch_size: Size of each processing batch
        remote: Whether to run model remotely
        patching_activations: tensor of shape(n_layers, n_heads, d_head)
        grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. Useful for interventions
        scaling_factor: scaling factor for the corruption head activation
        patch_pos: position to patch the corruption head activation. 
            -1: patch the last position
            "all": patch all positions

    Returns:
        base_ranks (torch.Tensor):
        base_probs (torch.Tensor):
        intervention_ranks (torch.Tensor):
        intervention_probs (torch.Tensor):
    """
    # Input validation
    total_samples = len(base_prompts)

    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]

    # Tokenize all data upfront
    #print("Tokenizing all data...")
    all_base_tokens = model.tokenizer(
        base_prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).input_ids
    base_token_len = all_base_tokens.shape[1]

    # Get token IDs for answers
    answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

    # Initialize results storage # TODO 
    base_ranks = []
    base_probs = []
    intervention_ranks = []
    intervention_probs = []

    # Process in batches
    # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start

        # Get current batch tokens from pre-tokenized data
        base_tokens = all_base_tokens[batch_start:batch_end]
        batch_answer_tokens = answer_tokens[batch_start:batch_end]

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)

        # Base run 
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(base_tokens):
                base_logits = model.output.logits[:,-1].save()

        # Run patching 
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(base_tokens):
                for layer in grouped_heads.keys():
                    # Get base attention output
                    if patch_pos == -1:
                        base_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                        base_reshaped = base_attn_output.reshape(current_batch_size, n_heads, d_head)

                        # patch activation for certain heads in this layer 
                        base_reshaped[:, grouped_heads[layer], :] = scaling_factor * patching_activations[layer, grouped_heads[layer]] 

                    elif patch_pos == "all":
                        base_attn_output = accessor.layers[layer].attention.output.unwrap().input
                        base_reshaped = base_attn_output.reshape(current_batch_size, base_token_len, n_heads, d_head)

                        # patch the activation for certain heads in this layer 
                        base_reshaped[:, :, grouped_heads[layer], :] = scaling_factor * patching_activations[layer, grouped_heads[layer]] 

                    else:
                        raise ValueError(f"Invalid patch position: {patch_pos}")

                interv_logits = model.output.logits[:,-1].save() #TODO: confirm this is right 

        batch_base_ranks, batch_base_probs = compute_token_rank_prob(base_logits, batch_answer_tokens)
        batch_intervention_ranks, batch_intervention_probs = compute_token_rank_prob(interv_logits, batch_answer_tokens)           

        base_ranks.append(batch_base_ranks)
        base_probs.append(batch_base_probs)
        intervention_ranks.append(batch_intervention_ranks)
        intervention_probs.append(batch_intervention_probs)
    
    base_ranks = torch.cat(base_ranks)
    base_probs = torch.cat(base_probs)
    intervention_ranks = torch.cat(intervention_ranks)
    intervention_probs = torch.cat(intervention_probs)

    return base_ranks, base_probs, intervention_ranks, intervention_probs


def eval_head_batch_residual_stream(
    model: LanguageModel, 
    model_name: str,
    base_prompts: list[str],
    answers: list[str],
    n_top_heads: int = 10,
    intervention_layer: int = None,
    batch_size: int = 8, remote: bool = False,
    seed: int = None,
    random_heads: bool = False,
    mean_activations: torch.Tensor = None,
    head_source: str = "EP",
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    #TODO: load layer from a saved map dict for each model 
    Evaluate the effect of adding or subtracting a vector to/from the residual stream. 
    This is used for evaluating the causal effect of FV heads.

    Args:
        model: Language model to analyze
        base_prompts: List of prompts
        corrupt_prompts: List of prompts
        answers: List of expected answer strings
        vector: vector to be added to or subtracted from the residual stream. 
            tensor of shape(1, d_model)
        intervention_layer: layer to intervene on. 
        batch_size: Size of each processing batch
        remote: Whether to run model remotely
        random_heads: whether to use random heads or the top heads of highest score 
        mean_activations: mean activations, tensor of shape(n_layers, n_heads, d_head)
        head_source: head extracted from "EP" or "IP" runs 
        scaling_factor: scaling factor for the corruption head activation

    Returns:
        base_ranks (torch.Tensor):
        base_probs (torch.Tensor):
        intervention_ranks (torch.Tensor):
        intervention_probs (torch.Tensor):
    """
    # Input validation
    total_samples = len(base_prompts)
    assert total_samples == len(answers)

    # Get vector 
    vector, _ = get_vector_from_universal_top_indirect_effect_heads(model, 
        model_name=model_name, n_top_heads=n_top_heads, mean_activations=mean_activations, 
        device="cuda:0", random_heads=random_heads, seed=seed, head_source=head_source)

    # Tokenize all data upfront
    #print("Tokenizing all data...")
    all_base_tokens = model.tokenizer(
        base_prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).input_ids
    base_token_len = all_base_tokens.shape[1]

    # Get token IDs for answers
    answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

    # Initialize results storage # TODO 
    base_ranks = []
    base_probs = []
    intervention_ranks = []
    intervention_probs = []

    # Process in batches
    # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)

        # Get current batch tokens from pre-tokenized data
        base_tokens = all_base_tokens[batch_start:batch_end]
        batch_answer_tokens = answer_tokens[batch_start:batch_end]

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)

        # Base run 
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(base_tokens):
                base_logits = model.output.logits[:,-1].save()

        # Run patching 
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(base_tokens):
                # Modify residual stream 
                accessor.layers[intervention_layer].unwrap().output[0][:,-1] += vector

                interv_logits = model.output.logits[:,-1].save() #TODO: confirm this is right 

        batch_base_ranks, batch_base_probs = compute_token_rank_prob(base_logits, batch_answer_tokens)
        batch_intervention_ranks, batch_intervention_probs = compute_token_rank_prob(interv_logits, batch_answer_tokens)           

        base_ranks.append(batch_base_ranks)
        base_probs.append(batch_base_probs)
        intervention_ranks.append(batch_intervention_ranks)
        intervention_probs.append(batch_intervention_probs)
    
    base_ranks = torch.cat(base_ranks)
    base_probs = torch.cat(base_probs)
    intervention_ranks = torch.cat(intervention_ranks)
    intervention_probs = torch.cat(intervention_probs)

    return base_ranks, base_probs, intervention_ranks, intervention_probs

def run_eval_EP_IP_shared_heads(model=None, model_name: str=None, d_name="country-capital",
    batch_size=8, remote=False, scaling_factor=1.0,
    EXP_SIZE=100, project_root: str = "/oscar/data/epavlick/zyang220/projects_2025/EP_IP",
    eval_heads: Union[list, dict]= None, 
    corruption_type="zs", 
    patch_pos: Union[int, str] = -1,
    intervention_type: str = "ablation", ):
    """
    #TODO: add pair-wise patching rather than patching the mean  
    Evaluate the effect of ablating or activating SHARED heads by patching the corrupted or clean activations. 
    For abaltion: use the same curroption for IP and EP so that patched activation is the same for both. 
    Args:
        eval_heads: list or dict
            list of heads of tuple (layer, head) or (layer, head, score)
            dict of grouped components keyed by layer {layer: [head1, head2, ...]}
        corruption_type: "zs" or "minimal"
        patch_pos: position to patch the corruption head activation. 
            -1: patch the last position
            "all": patch all positions #TODO: consider remove this option
        intervention_type: "ablation" or "activation"
    """
    if isinstance(eval_heads, list):
        grouped_heads = group_ranked_locations_by_layer(eval_heads)
    elif isinstance(eval_heads, dict):
        grouped_heads = eval_heads
    else:
        raise ValueError(f"Invalid eval_heads type: {type(eval_heads)}")

    # Get prompts
    ## get input and output token length  
    model_name = model_name.split("/")[-1]
    #TODO: might use a universal dataset_info file for all models 
    dataset_info = json.load(open(os.path.join(project_root, "datasets", "dataset_info", f"dataset_info_{model_name}.json")))
    INPUT_LENGTH = dataset_info[d_name]["input_length"]
    OUTPUT_LENGTH = dataset_info[d_name]["output_length"]
  
    IP_prompts, corrupt_prompts_IP, IP_answers = generate_instruction_prompts(
        d_name=d_name, 
        INPUT_LENGTH=INPUT_LENGTH, OUTPUT_LENGTH=OUTPUT_LENGTH, EXP_SIZE=EXP_SIZE, 
        model=model, instruction=dataset_info[d_name]["INST_prompt"],
        corruption_type=corruption_type, 
    )
    EP_prompts, corrupt_prompts_EP, EP_answers = generate_few_shot_prompts(
        d_name=d_name, 
        INPUT_LENGTH=INPUT_LENGTH, OUTPUT_LENGTH=OUTPUT_LENGTH, EXP_SIZE=EXP_SIZE, 
        model=model, corruption_type=corruption_type, 
    )

    # Get 1) corrupted or clean activation to patch later and 2) acc/ranks/probs of the run 
    if intervention_type == "ablation": # corruption run 
        corrupt_activation_IP, corrupt_ranks_IP, corrupt_probs_IP = get_head_activation_ranks_probs_mean(model, 
            corrupt_prompts_IP, IP_answers, batch_size=batch_size, remote=remote) # use the corrupt_prompts_IP for both IP and EP
    elif intervention_type == "activation": # clean run 
        clean_activation_IP, clean_ranks_IP, clean_probs_IP = get_head_activation_ranks_probs_mean(model, 
            IP_prompts, IP_answers, batch_size=batch_size, remote=remote
        )
        clean_activation_EP, clean_ranks_EP, clean_probs_EP = get_head_activation_ranks_probs_mean(model, 
            EP_prompts, EP_answers, batch_size=batch_size, remote=remote
        )
    else:
        raise ValueError(f"Invalid intervention type: {intervention_type}")

    # Run head eval for IP
    if intervention_type == "ablation":
        clean_ranks_IP, clean_probs_IP, intervention_ranks_IP, intervention_probs_IP = eval_head_batch(
            model, base_prompts=IP_prompts, answers=IP_answers, batch_size=batch_size, remote=remote, 
            patching_activations=corrupt_activation_IP, grouped_heads=grouped_heads,
            scaling_factor=scaling_factor, patch_pos=patch_pos,
        ) 
    elif intervention_type == "activation":
        corrupt_ranks_IP, corrupt_probs_IP, intervention_ranks_IP, intervention_probs_IP = eval_head_batch(
            model, base_prompts=corrupt_prompts_IP, answers=IP_answers, batch_size=batch_size, remote=remote, 
            patching_activations=clean_activation_IP, grouped_heads=grouped_heads,
            scaling_factor=scaling_factor, patch_pos=patch_pos, 
        ) 

    intervention_acc_IP = [compute_top_k_accuracy(
            intervention_ranks_IP.to('cpu'), k) for k in range(1,4)] 
    clean_acc_IP = [compute_top_k_accuracy(
            clean_ranks_IP.to('cpu'), k) for k in range(1,4)] 
    corrupt_acc_IP = [compute_top_k_accuracy(
                corrupt_ranks_IP.to('cpu'), k) for k in range(1,4)] 
    
    eval_dict_IP = {
        'clean_topk':clean_acc_IP, 
        'clean_ranks':clean_ranks_IP.tolist(), 
        'clean_probs':clean_probs_IP.tolist(), 
        'intervention_topk':intervention_acc_IP,
        'intervention_ranks':intervention_ranks_IP.tolist(), 
        'intervention_probs':intervention_probs_IP.tolist(),
        'corrupt_topk':corrupt_acc_IP,
        'corrupt_ranks':corrupt_ranks_IP.tolist(), 
        'corrupt_probs':corrupt_probs_IP.tolist(),
        }
    
    # Run head eval for EP
    if intervention_type == "ablation":
        clean_ranks_EP, clean_probs_EP, intervention_ranks_EP, intervention_probs_EP = eval_head_batch(
            model, EP_prompts, EP_answers, batch_size=batch_size, remote=remote, 
            patching_activations=corrupt_activation_IP, grouped_heads=grouped_heads,
            scaling_factor=scaling_factor, patch_pos=patch_pos, 
        )  # use the same corrupt_activation_IP for both IP and EP
    elif intervention_type == "activation":
        corrupt_ranks_EP, corrupt_probs_EP, intervention_ranks_EP, intervention_probs_EP = eval_head_batch(
            model, base_prompts=corrupt_prompts_EP, answers=EP_answers, batch_size=batch_size, remote=remote, 
            patching_activations=clean_activation_EP, grouped_heads=grouped_heads,
            scaling_factor=scaling_factor, patch_pos=patch_pos,
        ) 
    
    intervention_acc_EP = [compute_top_k_accuracy(
            intervention_ranks_EP.to('cpu'), k) for k in range(1,4)]    
    clean_acc_EP = [compute_top_k_accuracy(
            clean_ranks_EP.to('cpu'), k) for k in range(1,4)] 
    if intervention_type == "activation": # if ablation, use the corrupt_acc_IP for EP
        corrupt_acc_EP = [compute_top_k_accuracy(
            corrupt_ranks_EP.to('cpu'), k) for k in range(1,4)] 
    
    eval_dict_EP = {
        'clean_topk':clean_acc_EP, 
        'clean_ranks':clean_ranks_EP.tolist(), 
        'clean_probs':clean_probs_EP.tolist(), 
        'intervention_topk':intervention_acc_EP,
        'intervention_ranks':intervention_ranks_EP.tolist(), 
        'intervention_probs':intervention_probs_EP.tolist(),
        'corrupt_topk':corrupt_acc_EP if intervention_type == "activation" else corrupt_acc_IP,
        'corrupt_ranks':corrupt_ranks_EP.tolist() if intervention_type == "activation" else corrupt_ranks_IP.tolist(), 
        'corrupt_probs':corrupt_probs_EP.tolist() if intervention_type == "activation" else corrupt_probs_IP.tolist(),
        }

    return eval_dict_IP, eval_dict_EP

def Activate_FV_heads(corruption_type:str="zs", EXP_SIZE:int=100, 
    model:LanguageModel=None, model_name:str=None,
    project_root:str=None, save_root:str=None,
    d_name:str=None, FV_score_path:str=None,
    batch_size=8, remote=False, scaling_factor=1.0, patch_pos=-1,
    seed=42, n_top_heads=10, intervention_method="patch_head",
    intervention_layer:int=None, 
    mean_head_activations_root_ICL:str=None,
    mean_head_activations_root_INST:str=None,
):
    """
    #TODO: run IP heads eval 
    Args:
        intervention_method: "patch_head" or "add_2_residual_stream"
        intervention_layer: layer to intervene on if intervention_method == "add_2_residual_stream"
    """
    # Get prompts and answers 
    #TODO: might use a universal dataset_info file for all models 
    dataset_info = json.load(open(os.path.join(project_root, "datasets", "dataset_info", f"dataset_info_{model_name}.json")))
    INPUT_LENGTH = dataset_info[d_name]["input_length"]
    OUTPUT_LENGTH = dataset_info[d_name]["output_length"]
    _, corrupt_prompts, answers = generate_few_shot_prompts(d_name=d_name, model=model, 
    INPUT_LENGTH=INPUT_LENGTH, OUTPUT_LENGTH=OUTPUT_LENGTH, corruption_type=corruption_type,
    EXP_SIZE=EXP_SIZE, filter_correct=True)

    # Load mean head activation 
    EP_activations = torch.load(os.path.join(save_root,
        model_name, d_name, "Baseline", "EP", "mean_act_head_EP.pt"))
    IP_activations = torch.load(os.path.join(save_root,
        model_name, d_name, "Baseline", "IP", "mean_act_head_IP.pt"))
    # Load mean head act from function vector codebase 
    # mean_activations_path_ICL = os.path.join(mean_head_activations_root_ICL, 
    #         model_name.split('/')[-1], d_name, f'{d_name}_mean_head_activations.pt') 
    # mean_activations_path_INST = os.path.join(mean_head_activations_root_INST,
    #     model_name.split('/')[-1], d_name, f'{d_name}_mean_head_activations.pt') 
    # if os.path.exists(mean_activations_path_ICL):
    #     print(f"Loading Mean Activations from {mean_activations_path_ICL}")
    #     mean_head_acts_ICL = torch.load(mean_activations_path_ICL) #(Layers, Heads, grouped_tokens, head_dim) 
    #     print(mean_head_acts_ICL.shape)
    #     EP_activations = mean_head_acts_ICL[:, :, -1] 
    # else: 
    #     raise ValueError(f"Mean activations path {mean_activations_path_ICL} does not exist")
    # if os.path.exists(mean_activations_path_INST):
    #     print(f"Loading Mean Activations from {mean_activations_path_INST}")
    #     mean_head_acts_INST = torch.load(mean_activations_path_INST) #(Layers, Heads, grouped_tokens, head_dim) 
    #     print(mean_head_acts_INST.shape)
    #     IP_activations = mean_head_acts_INST[:, :, -1]  
    # else: 
    #     raise ValueError(f"Mean activations path {mean_activations_path_INST} does not exist")

    # Load and evaluate EP heads  
    fv_scores_file_path = os.path.join(FV_score_path, "function_vector",
        "fv_scores", f"{model_name}_fv_scores.json")
    with open(fv_scores_file_path, "r") as f:
        fv_scores = json.load(f)
    EP_grouped_heads = group_ranked_locations_by_layer(fv_scores["top_components"][:n_top_heads])
    print("EP_grouped_heads", EP_grouped_heads)

    ## Intervene with activation cached from EP runs 
    if intervention_method == "patch_head":
        _, _, ranks_EP_heads_EP_act, _ = eval_head_batch(
            model=model,
            base_prompts=corrupt_prompts,
            answers=answers,
            batch_size=batch_size,
            remote=remote,
            patching_activations=EP_activations, 
            grouped_heads=EP_grouped_heads, 
            scaling_factor=scaling_factor,
            patch_pos=patch_pos,
        )
    elif intervention_method == "add_2_residual_stream":
        _, _, ranks_EP_heads_EP_act, _ = eval_head_batch_residual_stream(
            model=model, model_name=model_name,
            base_prompts=corrupt_prompts, answers=answers,
            mean_activations=EP_activations, head_source="EP",
            batch_size=batch_size, 
            remote=remote, n_top_heads=n_top_heads,
            intervention_layer=intervention_layer,
            scaling_factor=scaling_factor,
            seed=seed, random_heads=False,
        )

    ## Intervene with activation cached from IP runs 
    if intervention_method == "patch_head":
        _, _, ranks_EP_heads_IP_act, _ = eval_head_batch(
            model=model,
            base_prompts=corrupt_prompts,
            answers=answers,
            batch_size=batch_size,
            remote=remote,
            patching_activations=IP_activations, 
            grouped_heads=EP_grouped_heads, 
            scaling_factor=scaling_factor,
            patch_pos=patch_pos,
        )
    elif intervention_method == "add_2_residual_stream":
        _, _, ranks_EP_heads_IP_act, _ = eval_head_batch_residual_stream(
            model=model, model_name=model_name,
            base_prompts=corrupt_prompts, answers=answers,
            mean_activations=IP_activations, head_source="EP",
            batch_size=batch_size,
            remote=remote, n_top_heads=n_top_heads,
            intervention_layer=intervention_layer,
            scaling_factor=scaling_factor,
            seed=seed,
            random_heads=False,
        )

    # Load and evaluate IP heads
    # fv_scores_file_path = os.path.join(FV_score_path, "function_vector_instructions",
    #     "fv_scores", f"{model_name}_fv_scores.json")
    # with open(fv_scores_file_path, "r") as f:
    #     fv_scores = json.load(f)
    # IP_grouped_heads = group_ranked_locations_by_layer(fv_scores["top_components"][:n_top_heads])

    # ## Intervene with activation cached from EP runs 
    # if intervention_method == "patch_head":
    #     _, _, ranks_IP_heads_EP_act, _ = eval_head_batch(
    #         model=model,
    #         base_prompts=corrupt_prompts,
    #         answers=answers,
    #         batch_size=batch_size,
    #         remote=remote,
    #         patching_activations=EP_activations, 
    #         grouped_heads=IP_grouped_heads, 
    #         scaling_factor=scaling_factor,
    #         patch_pos=patch_pos,
    #     )
    # elif intervention_method == "add_2_residual_stream":
    #     _, _, ranks_IP_heads_EP_act, _ = eval_head_batch_residual_stream(
    #         model=model, model_name=model_name,
    #         base_prompts=corrupt_prompts, answers=answers,
    #         mean_activations=EP_activations, head_source="IP",
    #         batch_size=batch_size, 
    #         remote=remote, n_top_heads=n_top_heads,
    #         intervention_layer=intervention_layer,
    #         scaling_factor=scaling_factor,
    #         seed=seed,
    #         random_heads=False,
    #     )
    # ## Intervene with activation cached from IP runs 
    # if intervention_method == "patch_head":
    #     _, _, ranks_IP_heads_IP_act, _ = eval_head_batch(
    #         model=model,
    #         base_prompts=corrupt_prompts,
    #         answers=answers,
    #         batch_size=batch_size,
    #         remote=remote,
    #         patching_activations=IP_activations, 
    #         grouped_heads=IP_grouped_heads, 
    #         scaling_factor=scaling_factor,
    #         patch_pos=patch_pos,
    #     )
    # elif intervention_method == "add_2_residual_stream":
    #     _, _, ranks_IP_heads_IP_act, _ = eval_head_batch_residual_stream(
    #         model=model, model_name=model_name,
    #         base_prompts=corrupt_prompts, answers=answers,
    #         mean_activations=IP_activations, head_source="IP",
    #         batch_size=batch_size, 
    #         remote=remote, n_top_heads=n_top_heads,
    #         intervention_layer=intervention_layer,
    #         scaling_factor=scaling_factor,
    #         seed=seed,
    #         random_heads=False,
    #     )

    # Sample and evaluate random heads 
    ## Intervene with activation cached from EP runs 
    random_heads = sample_random_heads(model,
        num_samples=n_top_heads, seed=seed)
    random_heads_grouped = group_ranked_locations_by_layer(random_heads)
    if intervention_method == "patch_head":
        _, _, ranks_random_heads_EP_act, _ = eval_head_batch(
            model=model,
            base_prompts=corrupt_prompts,
            answers=answers,
            batch_size=batch_size,
            remote=remote,
            patching_activations=EP_activations, 
            grouped_heads=random_heads_grouped, 
            scaling_factor=scaling_factor,
            patch_pos=patch_pos,
        )
    elif intervention_method == "add_2_residual_stream":
        _, _, ranks_random_heads_EP_act, _ = eval_head_batch_residual_stream(
            model=model, model_name=model_name,
            base_prompts=corrupt_prompts, answers=answers,
            mean_activations=EP_activations, 
            batch_size=batch_size, 
            remote=remote, n_top_heads=n_top_heads,
            intervention_layer=intervention_layer,
            scaling_factor=scaling_factor,
            seed=seed, random_heads=True,
        )
    ## Intervene with activation cached from IP runs 
    if intervention_method == "patch_head":
        _, _, ranks_random_heads_IP_act, _ = eval_head_batch(
            model=model,
            base_prompts=corrupt_prompts,
            answers=answers,
            batch_size=batch_size,
            remote=remote,  
            patching_activations=IP_activations, 
            grouped_heads=random_heads_grouped, 
            scaling_factor=scaling_factor,
            patch_pos=patch_pos,
        )
    elif intervention_method == "add_2_residual_stream":
        _, _, ranks_random_heads_IP_act, _ = eval_head_batch_residual_stream(
            model=model, model_name=model_name,
            base_prompts=corrupt_prompts, answers=answers,
            mean_activations=IP_activations, 
            batch_size=batch_size, 
            remote=remote, n_top_heads=n_top_heads,
            intervention_layer=intervention_layer,
            scaling_factor=scaling_factor,
            seed=seed, random_heads=True,
        )

    # Get acc and compile acc dict 
    acc_EP_heads_EP_act = [compute_top_k_accuracy(
            ranks_EP_heads_EP_act.to('cpu'), k) for k in range(1,4)] 
    acc_EP_heads_IP_act = [compute_top_k_accuracy(
            ranks_EP_heads_IP_act.to('cpu'), k) for k in range(1,4)] 
    # acc_IP_heads_EP_act = [compute_top_k_accuracy(
    #         ranks_IP_heads_EP_act.to('cpu'), k) for k in range(1,4)] 
    # acc_IP_heads_IP_act = [compute_top_k_accuracy(
    #         ranks_IP_heads_IP_act.to('cpu'), k) for k in range(1,4)] 
    acc_random_heads_EP_act = [compute_top_k_accuracy(
            ranks_random_heads_EP_act.to('cpu'), k) for k in range(1,4)] 
    acc_random_heads_IP_act = [compute_top_k_accuracy(
            ranks_random_heads_IP_act.to('cpu'), k) for k in range(1,4)] 
    
    acc_dict = {
        "acc_EP_heads_EP_act": acc_EP_heads_EP_act[0],
        "acc_EP_heads_IP_act": acc_EP_heads_IP_act[0],
        # "acc_IP_heads_EP_act": acc_IP_heads_EP_act,
        # "acc_IP_heads_IP_act": acc_IP_heads_IP_act,
        "acc_random_heads_EP_act": acc_random_heads_EP_act[0],
        "acc_random_heads_IP_act": acc_random_heads_IP_act[0],
        "random_heads": random_heads,
        "n_top_heads": n_top_heads,
    }

    # Save acc_dict to json file 
    save_path = os.path.join(save_root, model_name, d_name, 
        "Heads", "causal_intervention", "FV",
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 
        f"Activate_FV_in_{corruption_type}_runs_{intervention_method}.json"), "w") as f:
        json.dump(acc_dict, f)

