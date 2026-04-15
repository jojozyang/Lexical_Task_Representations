import os
import json
import argparse
from nnsight import LanguageModel
import torch

from Submodules.fix_incorrect_prompts import *

torch.set_grad_enabled(False)

if __name__ == "__main__":
    """
    Causal intervention to fix incorrect prompts or recover corrupted prompts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
        help="model name e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True,)
    parser.add_argument("--prompt_type_fix", type=str, required=True,
        help="prompt type used to fix the incorrect/corrupted prompts: EP or IP")
    parser.add_argument("--save_root", type=str, 
        default="../output",)
    parser.add_argument("--project_root", type=str, 
        default="../",
        help="directory of the codebase ")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--exp_size", type=int, default=100, help="number of examples to sample from the dataset")
    parser.add_argument("--average_size", type=int, default=100, help="number of prompts to average over")
    parser.add_argument("--intervention_component_prompt_type", type=str, 
        default="EP", help="prompt style used to identify the intervention component: EP or IP")
    parser.add_argument("--dataset_folder", type=str, 
        default="../datasets/abstractive", help="folder of the dataset")

    args = parser.parse_args()
    model_name = args.model_name
    d_name = args.d_name
    prompt_type_fix = args.prompt_type_fix
    save_root = args.save_root
    project_root = args.project_root
    batch_size = args.batch_size
    exp_size = args.exp_size
    average_size = args.average_size
    intervention_component_prompt_type = args.intervention_component_prompt_type
    dataset_folder = args.dataset_folder
    
    print("model_name", model_name)
    print("prompt_type_fix", prompt_type_fix)
    print("d_name", d_name)
    print("intervention_component_prompt_type", intervention_component_prompt_type)

    # Load model 
    model = LanguageModel(
        model_name,
        device_map="auto",
        dispatch=True,
    )
    model_name = model_name.split("/")[-1]

    # Parameters 
    scaling_factors = [-10, -6, -4, -2, 0, 1, 2, 3, 4, 5,6,7,8,9,10]
    print("scaling_factors", scaling_factors)
    print("batch_size", batch_size)
    print("exp_size", exp_size)
    print("average_size", average_size)

    if prompt_type_fix == "EP":
        prompt_index_list_fix = [1,5,10,20,30]
    elif prompt_type_fix == "IP":
        prompt_index_list_fix = [0,1,2,3,4]
    else:
        raise ValueError(f"prompt_type_fix {prompt_type_fix} not supported")

    # Load instruction_dict
    with open(os.path.join(project_root, "datasets", "dataset_info", 
        f"instruction_dict.json"), "r"
    ) as f:
        instruction_dict = json.load(f)

    control_map = {
    "country-capital": "park-country",
    "product-company": "park-country",
    "park-country": "country-currency",
    "landmark-country": "product-company",
    "country-currency": "park-country",
    "person-occupation": "person-sport",
    "person-sport": "person-occupation",
    "person-instrument": "person-sport",
    "antonym": "synonym",
    "synonym": "antonym",
    "singular-plural": "present-past",
    "present-past": "singular-plural",
    "next_item": "prev_item",
    "prev_item": "next_item",
    "english-french": "english-german",
    "english-german": "english-french",
    "english-spanish": "english-german",
    'landmark_country_capital': "person-sport",
    'park_country_capital': "person-sport",
    'product_company_ceo':'person-sport',
    'person_university_founder':'person-occupation',
    'person_university_year':'person-occupation',
    'antonym_french': "spanish",
    'antonym_spanish': "french",
    'antonym_german': "french",
}

    if intervention_component_prompt_type == "IP":
        ## load behavior results
        with open(os.path.join(save_root, model_name,
            "across_tasks", "Behavior", "IP_vary_n_inst_behavior.json"), "r") as f:
            result_dict_IP_tasks = json.load(f)
            
        ## get best acc index
            acc_list = [result_dict_IP_tasks[d_name][str(i)]['accuracy'] for i in range(5)]
            best_index = np.argmax(acc_list).item()
            intervention_component_prompt_index = best_index
    elif intervention_component_prompt_type == "EP":
        intervention_component_prompt_index = 5

    results = fix_incorrect_prompts_vary_prompt_instances_average_new(
        model=model, model_name=model_name, 
        prompt_type_2_be_fixed="EP", prompt_index_2_be_fixed=1,
        fix_component='Relation_grouped_heads', 
        intervention_component_prompt_type=intervention_component_prompt_type, 
        intervention_component_prompt_index=intervention_component_prompt_index,
        prompt_proportion=0.1, n_match=1,
        prompt_type_fix=prompt_type_fix, prompt_index_list_fix=prompt_index_list_fix,
        scaling_factors=scaling_factors, 
        target_task=d_name, control_d_name=control_map[d_name],
        save_root=save_root, instruction_dict=instruction_dict, 
        exp_size=exp_size, average_size=average_size, batch_size=batch_size,
        MAPS_save_path=os.path.join(save_root, model_name, d_name, 
            "Heads", "MAPS", "Relation_across_tasks"),
        save_results=True, save_path=os.path.join(save_root, model_name, d_name, 
            "Fix_incorrect_prompts", "Relation_across_tasks"),
        dataset_folder=dataset_folder,
        project_root=project_root,
    )
    print("Done")