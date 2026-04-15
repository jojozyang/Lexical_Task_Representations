import os
import json
import torch
from nnsight import LanguageModel
import pickle

from Shared_utils.prompt_utils import get_prompt_token, create_few_shot_prompts, create_instruction_prompts, get_index_from_behavior_dict, check_correctness
from Shared_utils.shared_utils import *
from Shared_utils.set_intervention import *
from Interface.Interface_utils import *
from Shared_utils.wrapper import get_accessor_config, get_model_specs, ModelAccessor


def fix_incorrect_prompts_head(
    model: LanguageModel=None,
    correct_prompt_tokens: torch.Tensor=None,
    incorrect_prompt_tokens: torch.Tensor=None,
    answer_tokens: torch.Tensor=None,
    grouped_heads: dict[int, list[int]]=None,
    batch_size: int=8,
    remote: bool=False,
    scale_factor: float=1.0,
) -> dict:
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]

    total_samples = correct_prompt_tokens.shape[0]
    #token_len = all_clean_tokens.shape[1]

    effects = []
    intervention_ranks = []
    incorrect_ranks = []
    correct_ranks = []

    # Process in batches
    #for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start
        
        # Get current batch tokens from pre-tokenized data
        batch_correct_tokens = correct_prompt_tokens[batch_start:batch_end]
        batch_incorrect_tokens = incorrect_prompt_tokens[batch_start:batch_end]
        batch_answer_tokens = answer_tokens[batch_start:batch_end]

        # Get incorrect logits at answer idx
        incorrect_logits = model.trace(batch_incorrect_tokens, trace=False, remote=remote)["logits"][:, -1]
        incorrect_answer_logits = incorrect_logits[torch.arange(current_batch_size), batch_answer_tokens]

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)

        head_activations_correct = {}  
        # Step 1: Collect correct activations and logits   
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(batch_correct_tokens):
                for layer in grouped_heads.keys():
                    attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
                    for head in grouped_heads[layer]:
                        head_activations_correct[(layer, head)] = attn_reshaped[:, head, :].clone().save()
                correct_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()
                correct_logits = model.output.logits[:,-1].save()

        # Step 2: Patch incorrect activations with correct activations
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(batch_incorrect_tokens):
                for layer in grouped_heads.keys():
                    attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
                    for head in grouped_heads[layer]:
                        attn_reshaped[:, head, :] = scale_factor * head_activations_correct[(layer, head)]
                interv_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()
                interv_logits = model.output.logits[:,-1].save() 

        # Calculate normalized intervention effect on logits
        intervention_diff = interv_answer_logits.to("cpu") - incorrect_answer_logits.to("cpu")
        baseline = correct_answer_logits.to("cpu") - incorrect_answer_logits.to("cpu")
        effect = intervention_diff / baseline 
        effects.append(effect)
        
        batch_intervention_ranks, _ = compute_token_rank_prob(interv_logits, batch_answer_tokens) 
        batch_incorrect_ranks, _ = compute_token_rank_prob(incorrect_logits, batch_answer_tokens)
        batch_correct_ranks, _ = compute_token_rank_prob(correct_logits, batch_answer_tokens)

        del interv_logits, incorrect_logits, correct_logits, head_activations_correct # free memory
        free_unused_cuda_memory()       
      
        intervention_ranks.append(batch_intervention_ranks.to('cpu'))
        incorrect_ranks.append(batch_incorrect_ranks.to('cpu'))
        correct_ranks.append(batch_correct_ranks.to('cpu'))

    effects = torch.cat(effects)
    logit_recovery = effects.mean()
    
    intervention_ranks = torch.cat(intervention_ranks)
    incorrect_ranks = torch.cat(incorrect_ranks)
    correct_ranks = torch.cat(correct_ranks)

    result_dict = {
        "intervention_ranks": intervention_ranks,
        "incorrect_ranks": incorrect_ranks,
        "logit_recovery": logit_recovery,
        "correct_ranks": correct_ranks,
    }

    return result_dict

def get_rank_acc_logit_dict(fix_result_dict):
    fix_interv_acc = compute_top_k_accuracy(fix_result_dict['intervention_ranks'].to('cpu'), 1)
    fix_incorrect_acc = compute_top_k_accuracy(fix_result_dict['incorrect_ranks'].to('cpu'), 1)
    fix_correct_acc = compute_top_k_accuracy(fix_result_dict['correct_ranks'].to('cpu'), 1)
    fix_interv_reciprocal_rank = [1 / (x+1) for x in fix_result_dict['intervention_ranks']]
    fix_interv_average_rr = sum(fix_interv_reciprocal_rank) / len(fix_interv_reciprocal_rank)
    fix_incorrect_reciprocal_rank = [1 / (x+1) for x in fix_result_dict['incorrect_ranks']]
    fix_incorrect_average_rr = sum(fix_incorrect_reciprocal_rank) / len(fix_incorrect_reciprocal_rank)

    assert fix_incorrect_acc == 0.0
    assert fix_correct_acc == 1.0

    rank_acc_logit_dict = {
        "interv_acc": fix_interv_acc.item(),
        "interv_rr": fix_interv_average_rr.item(),
        "incorrect_rr": fix_incorrect_average_rr.item(),
        "logit_recovery": fix_result_dict['logit_recovery'].item(),
    }
    return rank_acc_logit_dict

# def check_correctness(model, prompts, answers, 
#     batch_size=10, remote=False, query_index=None,
#     return_pred_tokens=False, return_answer_tokens=False
# ):
#     correct_index = []

#     prompt_tokens = model.tokenizer(
#         prompts, 
#         padding=True, 
#         padding_side="left", 
#         return_tensors="pt"
#     )["input_ids"]
#     answer_tokens = model.tokenizer(
#         answers,
#         add_special_tokens=False,
#         padding=True,
#         padding_side="right",
#         return_tensors="pt",
#     )["input_ids"][:, 0]
    
#     pred_tokens = []
#     #for i in tqdm(range(0, len(prompts), batch_size)):
#     for i in range(0, len(prompts), batch_size):
#         batch_prompt_tokens = prompt_tokens[i : i + batch_size]
#         batch_answer_tokens = answer_tokens[i : i + batch_size]
#         batch_pred_tokens = (
#             model.trace(batch_prompt_tokens, trace=False, remote=remote).logits[:, -1, :].argmax(dim=-1).cpu()
#         )
#         if return_pred_tokens:
#             pred_tokens.extend(batch_pred_tokens.tolist())

#         batch_correct = torch.where(batch_answer_tokens == batch_pred_tokens)[0].tolist()
#         correct_index += [idx + i for idx in batch_correct]

#     return_dict = {
#         "correct_index": correct_index,
#         "prompt_tokens": prompt_tokens,
#     }
#     #print(f"Accuracy: {len(correct_index)/len(prompts)} ({len(correct_index)}/{len(prompts)})")
#     if return_pred_tokens:
#         return_dict["pred_tokens"] = pred_tokens
#     if return_answer_tokens:
#         return_dict["answer_tokens"] = answer_tokens
#     return return_dict
        
def get_EP_IP_prompt_dict(model: LanguageModel,
    d_name: str = None, 
    batch_size: int=20, 
    return_pred_tokens: bool=False,
    dataset_info: dict=None,
    dataset_folder:str="../datasets/abstractive",
):
    """
    Run through all the prompts in the dataset. Note: Index are the index relative to the whole dataset.
    """
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
        dataset = json.load(f)

    # Get prompts 
    EP_prompts, EP_answers, _ = create_few_shot_prompts(
        dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":"
    )
    EP_prompt_dict = check_correctness(model=model, prompts=EP_prompts, answers=EP_answers, 
        batch_size=batch_size, return_pred_tokens=return_pred_tokens, return_answer_tokens=True
    )
    IP_prompts, IP_answers = create_instruction_prompts(
        dataset, instruction=dataset_info[d_name]["INST_prompt"],
    )
    IP_prompt_dict = check_correctness(model=model, prompts=IP_prompts, answers=IP_answers, 
        batch_size=batch_size, return_pred_tokens=return_pred_tokens, return_answer_tokens=False)
    
    EP_correct_index = EP_prompt_dict['correct_index']
    EP_prompt_tokens = EP_prompt_dict['prompt_tokens']
    IP_correct_index = IP_prompt_dict['correct_index']
    IP_prompt_tokens = IP_prompt_dict['prompt_tokens']
    answer_tokens = EP_prompt_dict['answer_tokens']

    # Check the overlap between EP_correct_index and IP_correct_index
    assert EP_prompt_tokens.shape[0] == IP_prompt_tokens.shape[0]
    both_correct_index = set(EP_correct_index) & set(IP_correct_index)
    both_correct_index = list(sorted(both_correct_index))

    # Only correct in EP 
    only_EP_correct_index = set(EP_correct_index) - set(IP_correct_index)
    only_EP_correct_index = list(sorted(only_EP_correct_index))

    # Only correct in IP 
    only_IP_correct_index = set(IP_correct_index) - set(EP_correct_index)
    only_IP_correct_index = list(sorted(only_IP_correct_index))

    prompt_dict = {
        "EP_prompt_tokens": EP_prompt_tokens,
        "IP_prompt_tokens": IP_prompt_tokens,
        "answer_tokens": answer_tokens,
        "EP_correct_index": EP_correct_index,
        "IP_correct_index": IP_correct_index,
        "only_EP_correct_index": only_EP_correct_index,
        "only_IP_correct_index": only_IP_correct_index,
        "both_correct_index": both_correct_index,
        "num_total_prompts": len(dataset),
        "num_prompts_4_correct_EP": len(only_IP_correct_index),
        "num_prompts_4_correct_IP": len(only_EP_correct_index),
        "EP_acc": len(EP_correct_index) / len(dataset),
        "IP_acc": len(IP_correct_index) / len(dataset),
    }
    if return_pred_tokens:
        prompt_dict["EP_pred_tokens"] = EP_prompt_dict['pred_tokens']
        prompt_dict["IP_pred_tokens"] = IP_prompt_dict['pred_tokens']
    return prompt_dict

def fix_incorrect_prompts_vary_prompt_instances_average(
    model:LanguageModel=None, model_name:str=None, 
    prompt_type_2_be_fixed:str=None, prompt_index_2_be_fixed:list=None,
    prompt_type_fix:str=None, prompt_index_list_fix:list=None,
    d_name:str=None, batch_size:int=None, 
    n_max_relation_heads:int=None, relation_head_threshold:float=None, 
    n_neurons_saved:int=None,
    save_results:bool=False, save_root:str=None,
    fix_component:str='Relation_grouped_heads',
    instruction_dict:dict=None,  scaling_factors:list=None, # either scaling factors or n_heads_list
    exp_size:int=100, average_size:int=100,
    FV_grouped_heads:dict={13: [27]}, control_d_name:str="english-german",
    n_execution_heads:int=None, 
):
    """
    # TODO: Add neurons 
    Fix incorrect prompts using the average activation from the correct prompts of different prompt instances 
        from either the same prompting style or a different prompting style. 
    
    NOTE: The collowing constrain is NOT applied: 
        The (query) index of prompts to be fixed should not be the same as the index of prompts fixing them 
        Reason: we want the prompt to be fixed to be the same for all fixing experiments!! 
    
    fix_component:"Relation_grouped_heads" or "FV_grouped_heads"
    prompt_index_2_be_fixed: the index of the prompt template or variant to be fixed
    average_size: number of prompts to average over to get the clean/fix activation 
    exp_size: number of prompts to fix 
    """
    n_heads = model.config.num_attention_heads
    # Components 
    if fix_component in ["Relation_grouped_heads", "Relation_neurons"]:
        component_dict = load_execution_relation_heads_neurons(d_name=d_name, 
            model_name=model_name, save_root=save_root, prompts_type=prompt_type_2_be_fixed,
            n_execution_heads=n_execution_heads, n_heads=n_heads,
            n_max_relation_heads=n_max_relation_heads, relation_head_threshold=relation_head_threshold,
            n_neurons_saved=n_neurons_saved,
        )
        if fix_component == "Relation_neurons":
            heads_or_neurons = "neurons"
            grouped_component_dict = group_ranked_locations_by_layer(component_dict[fix_component])
        else:
            heads_or_neurons = "heads"
            grouped_component_dict = component_dict[fix_component]

    elif fix_component == "FV_grouped_heads":
        heads_or_neurons = "heads"
        grouped_component_dict = FV_grouped_heads
    else:
        raise ValueError(f"fix_component {fix_component} not supported")
    
    # Prompts to be fixed (same for all fixing exp below)
    ## tokenize prompts 
    token_dict_2_be_fixed = get_prompt_token(model=model, d_name=d_name,
        prompt_type=prompt_type_2_be_fixed, 
        prompt_index=prompt_index_2_be_fixed, 
        instruction_dict=instruction_dict,
        return_answer_tokens=True,
    )
    prompt_token_2_be_fixed = token_dict_2_be_fixed["prompt_tokens"]
    answer_token_2_be_fixed = token_dict_2_be_fixed["answer_tokens"]
    ## get index of incorrect prompts 
    prompt_index_2_be_fixed_incorrect_index = get_index_from_behavior_dict(
        save_root=save_root, model_name=model_name,
        prompt_type=prompt_type_2_be_fixed, prompt_index=prompt_index_2_be_fixed,
        d_name=d_name, incorrect_or_correct="incorrect",
        dataset_size=prompt_token_2_be_fixed.shape[0],  
    )[:exp_size]
    
    # Fix incorrect prompts 
    result_dict = {}
    for prompt_index_fix in prompt_index_list_fix: 
        prompt_token_fix = get_prompt_token(model=model, d_name=d_name,
            prompt_type=prompt_type_fix, 
            prompt_index=prompt_index_fix, 
            instruction_dict=instruction_dict,
            return_answer_tokens=True,
        )["prompt_tokens"]
        prompt_index_fix_correct_index = get_index_from_behavior_dict(
            save_root=save_root, model_name=model_name,
            prompt_type=prompt_type_fix, prompt_index=prompt_index_fix,
            d_name=d_name, incorrect_or_correct="correct",
            dataset_size=prompt_token_fix.shape[0],   
        )
        # Filter to achieve the holdout effect: index of fix should not contain index of 2 be fixed
        prompt_index_fix_correct_index = set(prompt_index_fix_correct_index) - set(prompt_index_2_be_fixed_incorrect_index)
        prompt_index_fix_correct_index = sorted(list(prompt_index_fix_correct_index))[:average_size]

        # Run fixing experiment 
        assert len(prompt_index_2_be_fixed_incorrect_index) > 0
        if fix_component in ["Relation_grouped_heads", "FV_grouped_heads", "Relation_neurons"]:
            result_dict_exp = get_rank_logit_acc_raw_result_dict_scaling_average(
                model=model, batch_size=batch_size,
                all_clean_tokens=prompt_token_fix[prompt_index_fix_correct_index],
                all_corrupt_tokens=prompt_token_2_be_fixed[prompt_index_2_be_fixed_incorrect_index],
                all_answer_tokens=answer_token_2_be_fixed[prompt_index_2_be_fixed_incorrect_index],
                scaling_factors=scaling_factors, heads_or_neurons=heads_or_neurons,
                grouped_components=grouped_component_dict,
                prompt_index_list_2_be_fixed=prompt_index_2_be_fixed_incorrect_index,
                )
        else:
            raise ValueError(f"fix_component {fix_component} not supported")

        result_dict[f"rr_ranks_dict_{prompt_index_fix}"] = result_dict_exp['rank_dict']
        result_dict[f"interv_acc_dict_{prompt_index_fix}"] = result_dict_exp['acc_dict']
        result_dict[f"fixed_prompt_index_list_{prompt_index_fix}"] = result_dict_exp['fixed_prompt_index_list_dict']
    
    # Control
    prompt_token_control = get_prompt_token(model=model, d_name=control_d_name,
        prompt_type=prompt_type_fix, 
        prompt_index=1, 
        instruction_dict=instruction_dict,
    )["prompt_tokens"]
    if fix_component in ["Relation_grouped_heads", "FV_grouped_heads", "Relation_neurons"]:
        result_dict_control = get_rank_logit_acc_raw_result_dict_scaling_average(
                model=model, batch_size=batch_size,
                all_clean_tokens=prompt_token_control[prompt_index_fix_correct_index],
                all_corrupt_tokens=prompt_token_2_be_fixed[prompt_index_2_be_fixed_incorrect_index],
                all_answer_tokens=answer_token_2_be_fixed[prompt_index_2_be_fixed_incorrect_index],
                scaling_factors=scaling_factors, heads_or_neurons=heads_or_neurons,
                grouped_components=grouped_component_dict,
                prompt_index_list_2_be_fixed=prompt_index_2_be_fixed_incorrect_index,
        )
    else:
        raise ValueError(f"fix_component {fix_component} not supported")
    result_dict['rr_ranks_dict_control'] = result_dict_control['rank_dict']
    result_dict['interv_acc_dict_control'] = result_dict_control['acc_dict']
    result_dict['fixed_prompt_index_list_control'] = result_dict_control['fixed_prompt_index_list_dict']
    result_dict['exp_size'] = len(prompt_index_2_be_fixed_incorrect_index)

    
    # Save 
    result_dict['num_components_fix'] = get_component_count_in_grouped_component_dict(grouped_component_dict)
    result_dict['component_fix'] = grouped_component_dict
    if save_results:
        save_path = os.path.join(save_root, model_name, d_name, 
            "Fix_incorrect_prompts")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, 
            f"fix_{prompt_type_2_be_fixed}_{prompt_index_2_be_fixed}_with_{prompt_type_fix}_{fix_component}_scales_vary_prompt_instances_average.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(result_dict, f)
    
    return result_dict