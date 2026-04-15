import os
import pickle
from nnsight import LanguageModel
from Shared_utils.prompt_utils import *
from Shared_utils.shared_utils import *
from Shared_utils.set_intervention import *
#from Fix_incorrect_prompts.Fix_incorrect_prompts_utils import *

def fix_incorrect_prompts_vary_prompt_instances_average_new(
    model:LanguageModel=None, model_name:str=None, 
    prompt_type_2_be_fixed:str=None, prompt_index_2_be_fixed:list=None,
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
    dataset_folder:str="../datasets/abstractive",
    project_root:str=None, 
):
    """
    # TODO: Add neurons 
    Fix incorrect prompts using the average activation from the correct prompts of different prompt instances 
        from either the same prompting style or a different prompting style. 
    
    NOTE: The following constrain is NOT applied: 
        The (query) index of prompts to be fixed should not be the same as the index of prompts fixing them 
        Reason: we want the prompt to be fixed to be the same for all fixing experiments!! 
    
    fix_component:"Relation_grouped_heads" or "FV_grouped_heads"
    prompt_index_2_be_fixed: the index of the prompt template or variant to be fixed
    average_size: number of prompts to average over to get the clean/fix activation 
    exp_size: number of prompts to fix 
    """
    # Components 
    if intervention_component_list is None: 
        if dataset_folder == "../datasets/compositional":
            # load composition task list 
            with open(os.path.join(project_root, "datasets", 
                "dataset_info", f"compositional_task_dict.json"), "r") as f:
                compositional_task_dict = json.load(f)
            task_list = compositional_task_dict[target_task]
            for single_task in task_list:
                intervention_component_list = get_task_heads_list(
                    target_task=target_task, other_task=single_task,
                    prompt_type=intervention_component_prompt_type, 
                    prompt_template_index=intervention_component_prompt_index,
                    n_match=n_match, MAPS_score_threshold=prompt_proportion, 
                    correct_incorrect="correct", model_name=model_name,
                    save_root=save_root, save_path=MAPS_save_path, 
                )
                intervention_component_list.extend(intervention_component_list)
            intervention_component_list = list(set(intervention_component_list))
        else:
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
        dataset_folder=dataset_folder,
    )
    prompt_token_2_be_fixed = token_dict_2_be_fixed["prompt_tokens"]
    answer_token_2_be_fixed = token_dict_2_be_fixed["answer_tokens"]
    ## get index of incorrect prompts 
    prompt_index_2_be_fixed_incorrect_index = get_index_from_behavior_dict(
        save_root=save_root, model_name=model_name,
        prompt_type=prompt_type_2_be_fixed, prompt_index=prompt_index_2_be_fixed,
        d_name=target_task, incorrect_or_correct="incorrect",
        dataset_size=prompt_token_2_be_fixed.shape[0],  
    )[:exp_size]
    
    # Fix incorrect prompts 
    result_dict = {}
    for prompt_index_fix in prompt_index_list_fix: 
        prompt_token_fix = get_prompt_token(model=model, d_name=target_task,
            prompt_type=prompt_type_fix, 
            prompt_index=prompt_index_fix, 
            instruction_dict=instruction_dict,
            return_answer_tokens=True,
            dataset_folder=dataset_folder,
        )["prompt_tokens"]
        prompt_index_fix_correct_index = get_index_from_behavior_dict(
            save_root=save_root, model_name=model_name,
            prompt_type=prompt_type_fix, prompt_index=prompt_index_fix,
            d_name=target_task, incorrect_or_correct="correct",
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
    # prompt_token_control = get_prompt_token(model=model, d_name=control_d_name,
    #     prompt_type=prompt_type_fix, 
    #     prompt_index=1, 
    #     instruction_dict=instruction_dict,
    #     dataset_folder=dataset_folder,
    # )["prompt_tokens"]
    # prompt_index_fix_correct_index_control = get_index_from_behavior_dict(
    #     save_root=save_root, model_name=model_name,
    #     prompt_type=prompt_type_fix, prompt_index=1,
    #     d_name=control_d_name, incorrect_or_correct="correct",
    #     dataset_size=prompt_token_control.shape[0],   
    # )
    # if fix_component in ["Relation_grouped_heads", "FV_grouped_heads", "Relation_neurons"]:
    #     result_dict_control = get_rank_logit_acc_raw_result_dict_scaling_average(
    #             model=model, batch_size=batch_size,
    #             all_clean_tokens=prompt_token_control[prompt_index_fix_correct_index_control],
    #             all_corrupt_tokens=prompt_token_2_be_fixed[prompt_index_2_be_fixed_incorrect_index],
    #             all_answer_tokens=answer_token_2_be_fixed[prompt_index_2_be_fixed_incorrect_index],
    #             scaling_factors=scaling_factors, heads_or_neurons=heads_or_neurons,
    #             grouped_components=grouped_component_dict,
    #             prompt_index_list_2_be_fixed=prompt_index_2_be_fixed_incorrect_index,
    #     )
    # else:
    #     raise ValueError(f"fix_component {fix_component} not supported")
    # result_dict['rr_ranks_dict_control'] = result_dict_control['rank_dict']
    # result_dict['interv_acc_dict_control'] = result_dict_control['acc_dict']
    # result_dict['fixed_prompt_index_list_control'] = result_dict_control['fixed_prompt_index_list_dict']
    # result_dict['exp_size'] = len(prompt_index_2_be_fixed_incorrect_index)

    
    # Save 
    result_dict['num_components_fix'] = get_component_count_in_grouped_component_dict(grouped_component_dict)
    result_dict['component_fix'] = grouped_component_dict
    if save_results:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = (
            f"fix_{prompt_type_2_be_fixed}_{prompt_index_2_be_fixed}_"
            f"with_{intervention_component_prompt_type}{intervention_component_prompt_index}_"
            f"{fix_component}_n-match{n_match}_prompt-p{prompt_proportion}_"
            f"act_{prompt_type_fix}_"
            "holdout_average.pkl"
            )
        save_file = os.path.join(save_path, filename)
        with open(save_file, "wb") as f:
            pickle.dump(result_dict, f)
    
    return result_dict