import os
import pickle
from nnsight import LanguageModel
from Shared_utils.shared_utils import *
from Shared_utils.set_intervention import *
from Shared_utils.prompt_utils import *
torch.set_grad_enabled(False)

def fix_corrupted_prompts_vary_prompt_instances_average_new(
    model:LanguageModel=None, model_name:str=None, 
    prompt_type_2_be_fixed:str="zs",  prompt_index_2_be_fixed=1,
    prompt_type_fix:str=None, prompt_index_list_fix:list=None,
    target_task:str=None, 
    intervention_component_prompt_type:str=None, intervention_component_prompt_index:int=None,
    prompt_proportion:float=None,  n_match:int=None,
    save_results:bool=False, save_root:str=None, MAPS_save_path:str=None,
    fix_component:str='Relation_grouped_heads',
    instruction_dict:dict=None,  scaling_factors:list=None, # either scaling factors or n_heads_list
    exp_size:int=100, average_size:int=100,
    control_d_name:str="english-german", batch_size:int=None,
    heads_or_neurons:str="heads", save_path:str=None,
    intervention_component_list:list=None,
    run_control:bool=True, run_exp:bool=True,
):
    """
    # TODO: Add neurons 
    Fix incorrect prompts using the average activation from the correct prompts of different prompt instances 
        from either the same prompting style or a different prompting style. 
    
    NOTE: The collowing constrain is NOT applied: 
        The (query) index of prompts to be fixed should not be the same as the index of prompts fixing them 
        Reason: we want the prompt to be fixed to be the same for all fixing experiments!! 
    
    fix_component:"Relation_grouped_heads" or "FV_grouped_heads"
    prompt_type_2_be_fixed: "zs" zero-shot prompts
    prompt_index_2_be_fixed: the index of the prompt template or variant to be fixed
    average_size: number of prompts to average over to get the clean/fix activation 
    exp_size: number of prompts to fix 
    """

    # Components 
    if intervention_component_list is None: 
        intervention_component_list = get_task_heads_list(
            target_task=target_task, other_task=target_task,
            prompt_type=intervention_component_prompt_type, 
            prompt_template_index=intervention_component_prompt_index,
            n_match=n_match, MAPS_score_threshold=prompt_proportion, 
            correct_incorrect="correct", model_name=model_name,
            save_root=save_root, save_path=MAPS_save_path, 
        )
        
    grouped_component_dict = group_ranked_locations_by_layer(intervention_component_list)
    
    # Prompts to be fixed (same for all fixing exp below)
    ## tokenize prompts 
    token_dict_2_be_fixed = get_prompt_token(model=model, d_name=target_task,
        prompt_type=prompt_type_2_be_fixed, 
        prompt_index=prompt_index_2_be_fixed,
        instruction_dict=instruction_dict,
        return_answer_tokens=True,
    )
    prompt_token_2_be_fixed = token_dict_2_be_fixed["prompt_tokens"]
    answer_token_2_be_fixed = token_dict_2_be_fixed["answer_tokens"]
    ## get index of incorrect prompts 
    prompt_index_2_be_fixed_index = get_index_from_behavior_dict(
        save_root=save_root, model_name=model_name,
        prompt_type="EP", prompt_index=prompt_index_2_be_fixed,
        d_name=target_task, incorrect_or_correct="correct",
        dataset_size=prompt_token_2_be_fixed.shape[0],  
    )[:exp_size]
    
    # Fix incorrect prompts 
    result_dict = {}
    if run_exp:
        for prompt_index_fix in prompt_index_list_fix: 
            prompt_token_fix = get_prompt_token(model=model, d_name=target_task,
                prompt_type=prompt_type_fix, 
                prompt_index=prompt_index_fix, 
                instruction_dict=instruction_dict,
                return_answer_tokens=True,
            )["prompt_tokens"]
            prompt_index_fix_correct_index = get_index_from_behavior_dict(
                save_root=save_root, model_name=model_name,
                prompt_type=prompt_type_fix, prompt_index=prompt_index_fix,
                d_name=target_task, incorrect_or_correct="correct",
                dataset_size=prompt_token_fix.shape[0],   
            )
            # Filter to achieve the holdout effect: index of fix should not contain index of 2 be fixed
            prompt_index_fix_correct_index = set(prompt_index_fix_correct_index) - set(prompt_index_2_be_fixed_index)
            prompt_index_fix_correct_index = sorted(list(prompt_index_fix_correct_index))[:average_size]

            # Run fixing experiment 
            assert len(prompt_index_2_be_fixed_index) > 0
            if fix_component in ["Relation_grouped_heads", "FV_grouped_heads", "Relation_neurons"]:
                result_dict_exp = get_rank_logit_acc_raw_result_dict_scaling_average(
                    model=model, batch_size=batch_size,
                    all_clean_tokens=prompt_token_fix[prompt_index_fix_correct_index],
                    all_corrupt_tokens=prompt_token_2_be_fixed[prompt_index_2_be_fixed_index],
                    all_answer_tokens=answer_token_2_be_fixed[prompt_index_2_be_fixed_index],
                    scaling_factors=scaling_factors, heads_or_neurons=heads_or_neurons,
                    grouped_components=grouped_component_dict,
                    prompt_index_list_2_be_fixed=prompt_index_2_be_fixed_index,
                    )
            else:
                raise ValueError(f"fix_component {fix_component} not supported")

            result_dict[f"rr_ranks_dict_{prompt_index_fix}"] = result_dict_exp['rank_dict']
            result_dict[f"interv_acc_dict_{prompt_index_fix}"] = result_dict_exp['acc_dict']
            result_dict[f"fixed_prompt_index_list_{prompt_index_fix}"] = result_dict_exp['fixed_prompt_index_list_dict']
    
    # Control
    if run_control:
        prompt_token_control = get_prompt_token(model=model, d_name=control_d_name,
            prompt_type="EP", 
            prompt_index=1, 
            instruction_dict=instruction_dict,
        )["prompt_tokens"]
        prompt_index_fix_correct_index_control = get_index_from_behavior_dict(
            save_root=save_root, model_name=model_name,
            prompt_type=prompt_type_fix, prompt_index=1,
            d_name=control_d_name, incorrect_or_correct="correct",
            dataset_size=prompt_token_control.shape[0],   
        )
        if fix_component in ["Relation_grouped_heads", "FV_grouped_heads", "Relation_neurons"]:
            result_dict_control = get_rank_logit_acc_raw_result_dict_scaling_average(
                    model=model, batch_size=batch_size,
                    all_clean_tokens=prompt_token_control[prompt_index_fix_correct_index_control],
                    all_corrupt_tokens=prompt_token_2_be_fixed[prompt_index_2_be_fixed_index],
                    all_answer_tokens=answer_token_2_be_fixed[prompt_index_2_be_fixed_index],
                    scaling_factors=scaling_factors, heads_or_neurons=heads_or_neurons,
                    grouped_components=grouped_component_dict,
                    prompt_index_list_2_be_fixed=prompt_index_2_be_fixed_index,
            )
        else:
            raise ValueError(f"fix_component {fix_component} not supported")
        result_dict['rr_ranks_dict_control'] = result_dict_control['rank_dict']
        result_dict['interv_acc_dict_control'] = result_dict_control['acc_dict']
        result_dict['fixed_prompt_index_list_control'] = result_dict_control['fixed_prompt_index_list_dict']
        result_dict['exp_size'] = len(prompt_index_2_be_fixed_index)
    
    # Save 
    result_dict['num_components_fix'] = len(intervention_component_list)
    result_dict['component_fix'] = intervention_component_list
    if save_results:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = (
            f"fix_{prompt_type_2_be_fixed}_"
            f"with_{intervention_component_prompt_type}{intervention_component_prompt_index}_"
            f"{fix_component}_n-match{n_match}_prompt-p{prompt_proportion}_"
            f"act_{prompt_type_fix}_"
            "holdout_average.pkl"
            )
        save_file = os.path.join(save_path, filename)
        with open(save_file, "wb") as f:
            pickle.dump(result_dict, f)
    
    return result_dict

def get_save_retreival_heads_causal_corruption_results(
    model:LanguageModel=None, model_name:str=None,
    d_name=None, prompt_corruption_type="random_query",
    save_root=None, 
    prompt_type_2_be_fixed=None, prompt_index_2_be_fixed=None,
    prompt_type_fix=None,  prompt_index_list_fix=None, fix_component="Retrieval_heads",
    intervention_component_prompt_type=None, intervention_component_prompt_index=None,
    n_match=None, n_heads_list=None, 
    exp_size=None, batch_size=None, 
    save_results=True, save_path=None,
    instruction_dict=None,
    d_control_name="capitalize",
):
    """
    Args:
        n_heads: int, number of heads in the model 
    """
    #  Get a dict of heads to intervene, key is the number of heads to intervene 
    top_heads_dict = {}
    ranked_heads = get_ranked_heads_from_MAPS_score(
        target_task=d_name, other_task=d_name,
        prompt_type=intervention_component_prompt_type, 
        prompt_template_index=intervention_component_prompt_index,
        correct_incorrect="correct", model_name=model_name,
        save_root=save_root, save_path=None, 
        component_type="Retrieval", n_match=n_match,
    )
    for top_n_heads in n_heads_list:
        grouped_heads = group_ranked_locations_by_layer(ranked_heads[:top_n_heads])
        top_heads_dict[top_n_heads] = grouped_heads
    
    # Corrupted prompts to be fixed 
    token_dict_2_be_fixed = get_prompt_token(model=model, d_name=d_name,
        prompt_type=prompt_type_2_be_fixed, 
        prompt_index=prompt_index_2_be_fixed,
        instruction_dict=instruction_dict,
        corrupt_type=prompt_corruption_type,
        return_answer_tokens=False,
    )
    prompt_token_2_be_fixed = token_dict_2_be_fixed["prompt_tokens"][:exp_size]

    # Fix corrupted prompts 
    result_dict = {}
    for prompt_index_fix in prompt_index_list_fix: 
        token_dict_fix = get_prompt_token(model=model, d_name=d_name,
            prompt_type=prompt_type_fix, 
            prompt_index=prompt_index_fix, 
            instruction_dict=instruction_dict,
            return_answer_tokens=True,
        )
        prompt_token_fix = token_dict_fix["prompt_tokens"][:exp_size]
        answer_tokens= token_dict_fix["answer_tokens"][:exp_size]

        rr_ranks_dict, logit_recovery_dict, interv_acc_dict = get_rank_logit_acc_raw_result_dict(
            model=model, 
            top_heads=top_heads_dict,
            all_clean_tokens=prompt_token_fix,
            all_corrupt_tokens=prompt_token_2_be_fixed,
            all_answer_tokens=answer_tokens, 
            batch_size=batch_size,
            intervention_type="set_component_to_logits",
        )
        
        result_dict[f"rr_ranks_dict_{prompt_index_fix}"] = rr_ranks_dict
        result_dict[f"interv_acc_dict_{prompt_index_fix}"] = interv_acc_dict
        result_dict[f"logit_recovery_dict_{prompt_index_fix}"] = logit_recovery_dict

    # control
    # IP_prompts_control, _, _ = generate_instruction_prompts(
    #     d_name=d_control_name, EXP_SIZE=exp_size, model=model, 
    #     INPUT_LENGTH=dataset_info[d_control_name]["input_length"], 
    #     OUTPUT_LENGTH=dataset_info[d_control_name]["output_length"],
    #     corruption_type=prompt_corruption_type, 
    #     instruction=dataset_info[d_control_name]["INST_prompt"],
    # )
    # control_rr_ranks_dict, control_logit_recovery_dict, control_interv_acc_dict = get_rank_logit_acc_raw_result_dict(
    #         model=model, 
    #         top_heads=top_heads,
    #         prompts=IP_prompts_control,
    #         corrupt_prompts=EP_corrupt_prompts if prompts_type == "EP" else IP_corrupt_prompts,
    #         answers=IP_answers, batch_size=batch_size,
    #         intervention_type="set_component_to_logits",
    #     )

    # result_dict["rr_ranks_dict_control"] = control_rr_ranks_dict
    # result_dict["interv_acc_dict_control"] = control_interv_acc_dict
    # result_dict["logit_recovery_dict_control"] = control_logit_recovery_dict

   
    if save_results: 
        # Save results 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = (
            f"fix_{prompt_type_2_be_fixed}{prompt_index_2_be_fixed}_"
            f"with_{intervention_component_prompt_type}{intervention_component_prompt_index}_"
            f"{fix_component}_n-match{n_match}_"
            f"act_{prompt_type_fix}_"
            "holdout.pkl"
            )
        file_path = os.path.join(save_path, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(result_dict, f)
        
    return result_dict
