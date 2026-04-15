from datasets import load_dataset
from huggingface_hub import hf_hub_download
import argparse
import os
import pickle
import numpy as np
import torch

from nnsight import LanguageModel

from Shared_utils.shared_utils import *
from Submodules.identify_heads_early_decode import *
torch.set_grad_enabled(False)

def get_save_MAPS_scores_across_tasks_nnsight_generation(
    model:LanguageModel, prompts:list, 
    d_name:str, n_match_list:list, 
    component_type:str="relation", n:int=None, 
    k:int=10, k_list:list=None, 
    task_list:list=None,
    save:bool=True, save_path:str=None,
    batch_size:int=10, task_relation_dict:dict=None,
):
    # Model specs 
    spec = get_model_specs(model)
    n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]

    if component_type == "Relation":
        relation = True
        component_name = "lexical_task_heads"
    else:
        relation = False

    # --- Get prompt tokens ---
    all_prompt_tokens = model.tokenizer(
        prompts, 
        padding=True, 
        padding_side="left", 
        return_tensors="pt"
    )["input_ids"]

    all_answer_tokens = None 

    # --- Cache heads'output & calculate MAPS scores ---
    num_examples = all_prompt_tokens.shape[0]
    #print("num of examples: ", num_examples)
    results = {}
    if n_match_list is not None:
        for n_match in n_match_list:
            results[n_match] = {k: np.empty((num_examples, n_layers, n_heads)) for k in task_list}
    elif k_list is not None:
        for k in k_list:
            results[k] = {t: np.empty((num_examples, n_layers, n_heads)) for t in task_list}
    else:
        raise ValueError("Either n_match_list or k_list must be provided")

    _, cached_act = cache_act_nnsight(model=model, 
            cache_component_type="heads", 
            all_tokens=all_prompt_tokens,
            all_answer_tokens=all_answer_tokens,
            remote=False, batch_size=batch_size,
        )
    if n_match_list is not None:
        for n_match in n_match_list:
            for task in task_list:
                if relation:
                    # Convert candidate words to token IDs for each task ---
                    candidate_token_ids, _ = convert_expand_candidate_strs_to_token_ids_nnsight(
                        model, 
                        candidate_strs=task_relation_dict[task],
                        output_str_tokens=True
                    )
                    # convert to tensor 
                    candidate_token_ids = torch.tensor(candidate_token_ids)
                else:
                    candidate_token_ids = all_answer_tokens

                metrics_raw = calculate_component_metrics_from_heads_output(
                    model=model, heads_output=cached_act, 
                    dst_tokens=candidate_token_ids,
                    apply_ln=True, k=k, n_match=n_match, 
                    component_type=component_name
                )

                results[n_match][task] = metrics_raw["MAPS_scores"]
    elif k_list is not None:
        for k in k_list:
            for task in task_list:
                if relation:
                    # Convert candidate words to token IDs for each task ---
                    candidate_token_ids, candidate_strs_set = convert_expand_candidate_strs_to_token_ids_nnsight(
                        model, 
                        candidate_strs=task_relation_dict[d_name],
                        output_str_tokens=True
                    )
                    print("candidate_strs_set", candidate_strs_set)
                    # convert to tensor 
                    candidate_token_ids = torch.tensor(candidate_token_ids)
                else:
                    candidate_token_ids = all_answer_tokens

                metrics_raw = calculate_component_metrics_from_heads_output(
                    model=model, heads_output=cached_act, 
                    dst_tokens=candidate_token_ids,
                    apply_ln=True, k=k, n_match=n_match, 
                    component_type=component_name
                )

                results[k][task] = metrics_raw["MAPS_scores"]
    else:
        raise ValueError("Either n_match_list or k_list must be provided")

    #print(f"results_{n_match}_{task}.shape", results[n_match][task].shape)

    if save:
        if save_path is None:
            save_path = os.path.join(save_root, model_name, d_name,
                "Heads", "MAPS", f"{component_type}_across_tasks",)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #print(save_path)
        with open(os.path.join(save_path, 
            f"{d_name}_MAPS_{component_type}_heads_across_tasks.pkl"),
        "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    """
    Identify lexical task heads or retrieval heads
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
        help="model name e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True,)
    parser.add_argument("--component_type", type=str, required=True, 
        help="component type: lexical_task or Retrieval heads")
    parser.add_argument("--monitor_other_task", type=bool, default=False, 
        help="whether to monitor other task's lexical task heads while running the prompts for a giventarget task")
    parser.add_argument("--save_root", type=str, 
        default="../output",)
    parser.add_argument("--project_root", type=str, 
        default="../",
        help="directory of the codebase ")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--k", type=int, default=10, help="top k decoded tokens to look for match")
    parser.add_argument("--exp_size", type=int, default=100, help="number of examples to sample from the dataset")

    args = parser.parse_args()
    model_name = args.model_name
    save_root = args.save_root
    project_root = args.project_root
    d_name = args.d_name
    monitor_other_task = args.monitor_other_task
    batch_size = args.batch_size
    k = args.k
    exp_size = args.exp_size
    component_type = args.component_type
    if component_type == "lexical_task":
        component_type = "Relation"

    print("d_name", d_name)

    # Parameters 
    # n_match_list=[1]
    # k_list=None
    # n_match = None
    
    n_match_list = None
    k_list=[20,30]
    n_match = 1

    if monitor_other_task:
        task_list = ["python", "js"]
    else: 
        task_list = [d_name]

    task_relation_dict = {
        "python": ["python"],
        "js": ["JavaScript"]
    }

    # Load model 
    model = LanguageModel(
        model_name,
        device_map="auto",
        dispatch=True
    )
    model_name = model_name.split("/")[-1]

    # Dataset 
    local_file_path = hf_hub_download(
    repo_id="zai-org/humaneval-x",
    filename=f"data/{d_name}/data/humaneval.jsonl",
    repo_type="dataset"
    )
    ds = load_dataset("json", data_files=local_file_path, split="train")

    # Generate prompts 
    prompts = []
    for i in range(len(ds)):
        prompts.append(ds[i]["prompt"])
    prompts = prompts[:exp_size]

    get_save_MAPS_scores_across_tasks_nnsight_generation(
    model=model, prompts=prompts, 
    d_name=d_name, task_list=task_list,
    component_type=component_type, 
    batch_size=batch_size,
    n_match_list=n_match_list, n=n_match, 
    k_list=k_list, k=k, 
    save=True, task_relation_dict=task_relation_dict,
    )