from datasets import load_dataset
from huggingface_hub import hf_hub_download
import argparse
from nnsight import LanguageModel

from Shared_utils.shared_utils import *
from Submodules.identify_heads_early_decode import *
from Shared_utils.set_intervention import *

torch.set_grad_enabled(False)

def intervene_generation(
    model:LanguageModel, 
    convert_2_task:str, 
    original_task:str, 
    gen_prompt_type:str,    
    exp_size:int, 
    n_generate_tokens:int, 
    intervene_positions:list[int], 
    scaling_factor:list[float], 
    batch_size:int=64, 
    component_type:str="Relation", 
    k:int=10, MAPS_score_threshold=0.1,
    save_root:str=None, task_relation_dict:dict=None,
    k_list:list[int]=None,
):
    # --- Generate prompts ---
    ## prompts patch from 
    task = convert_2_task
    if task == "js":
        code_name = "JavaScript"
    else:
        code_name = task
    prompts_patch_from = [f"Write {code_name} code.", f"Write {code_name} code." 
        f"Please write {code_name} code to solve the problem.", 
        f"Help me write {code_name} code to solve the problem.",
        f"Generate {code_name} code to solve the problem."]
    print("prompts_patch_from: ", prompts_patch_from[0])

    ## prompts patch to 
    task = original_task
    if task == "js":
        code_name = "JavaScript"
    else:
        code_name = task
    code_instruction  = f"Write {code_name} code for the following problem.\n\n"
    local_file_path = hf_hub_download(
        repo_id="zai-org/humaneval-x",
        filename=f"data/{task}/data/humaneval.jsonl",
        repo_type="dataset"
    )
    ds = load_dataset("json", data_files=local_file_path, split="train")
    if gen_prompt_type == "problem":
        prompts_patch_to = ds['prompt'][:exp_size]
    elif gen_prompt_type == "INST_problem":
        prompts_patch_to = [code_instruction+ds[i]['prompt'] for i in range(len(ds))][:exp_size]
    else:
        raise ValueError("Invalid gen_prompt_type")

    print("prompts_patch_to: ", prompts_patch_to[0])
    prompts_id_patch_to = [ds[i]['task_id'] for i in range(exp_size)]

    # Find lexical task heads 
    task = convert_2_task
    task = "js"
    if task == "js":
        code_name = "JavaScript"
    else:
        code_name = task
    prompts_4_heads = [f"Write {code_name} code.", f"Write {code_name} code." 
        f"Please write {code_name} code to solve the problem.", 
        f"Help me write {code_name} code to solve the problem.",
        f"Generate {code_name} code to solve the problem."]

    # Load or generate heads 
    if gen_prompt_type in ["problem"]:
        save_path = os.path.join(save_root, model_name, convert_2_task,
            "Heads", "MAPS", f"{component_type}_across_tasks",)
    elif gen_prompt_type == "INST_problem":
        save_path = os.path.join(save_root, model_name, convert_2_task,
            "Heads", "MAPS", f"{component_type}_across_tasks_INST",)
    else:
        raise ValueError("Invalid gen_prompt_type")

    if not os.path.exists(save_path):
        print(f"Detecting {convert_2_task} heads")
        get_save_MAPS_scores_across_tasks_nnsight_generation(
            model=model, model_name=model_name, prompts=prompts_4_heads, 
            d_name=convert_2_task, task_list=[convert_2_task],
            component_type="Relation", 
            batch_size=batch_size,
            n_match_list=None, n_match=1, 
            k_list=k_list, k=None, 
            save=True, task_relation_dict=task_relation_dict,
            save_root=save_root,
        )
    
    print(f"Loading {convert_2_task} heads")
    list_of_heads = get_MAPS_head(
        model_name=model_name,
        target_task=convert_2_task,
        other_task=convert_2_task,
        k_or_n_match=k,
        component_type="Relation",
        MAPS_score_threshold=MAPS_score_threshold,
        save_path=save_path, save_root=save_root,
    )
    print(f"num of {convert_2_task} heads to intervene on: {len(list_of_heads)}")

    # Covert heads to grouped components 
    grouped_component_dict = group_ranked_locations_by_layer(list_of_heads)

    interv_generation = set_activation_patching_generation(
        model=model, batch_size=batch_size,
        grouped_components=grouped_component_dict,
        prompts_patch_from=prompts_patch_from,
        prompts_patch_to=prompts_patch_to,
        remote=False,
        scaling_factor=scaling_factor,
        n_generate_tokens=n_generate_tokens,
        intervene_positions=intervene_positions,  # generation step indices (0-indexed) to apply patch
    )

    result_dict = {
        "prompts_id_patch_to": prompts_id_patch_to,
        "prompts_patch_to": prompts_patch_to,
        "interv_generation": interv_generation,
        "prompts_patch_from": prompts_patch_from,
    }

    return result_dict

if __name__ == "__main__":
    """
    Identify lexical task heads or retrieval heads
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
        help="model name e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--original_task", type=str, required=True,)
    parser.add_argument("--convert_2_task", type=str, required=True,)
    parser.add_argument("--scaling_factor", type=float, required=True,)
    parser.add_argument("--save_root", type=str, 
        default="../output",)
    parser.add_argument("--project_root", type=str, 
        default="../",
        help="directory of the codebase ")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--exp_size", type=int, default=128, help="number of examples to sample from the dataset")
    parser.add_argument("--n_generate_tokens", type=int, default=256, help="number of tokens to generate")

    args = parser.parse_args()
    model_name = args.model_name
    original_task = args.original_task
    convert_2_task = args.convert_2_task
    batch_size = args.batch_size
    save_root = args.save_root
    project_root = args.project_root
    scaling_factor = args.scaling_factor
    exp_size = args.exp_size
    n_generate_tokens = args.n_generate_tokens

    print("model_name", model_name)
    print("original_task", original_task)
    print("convert_2_task", convert_2_task)
    print("batch_size", batch_size)
    print("save_root", save_root)
    print("project_root", project_root)
    print("scaling_factor", scaling_factor)
    print("exp_size", exp_size)
    print("n_generate_tokens", n_generate_tokens)
    # Load model 
    model = LanguageModel(
        model_name,
        device_map="auto",
        dispatch=True,
    )
    model_name = model_name.split("/")[-1]

    # Parameters 
    exp_size = exp_size
    n_generate_tokens = n_generate_tokens
    intervene_positions=[i for i in range(n_generate_tokens)]
    max_terv_pos = max(intervene_positions)
    k_list = [10, 20, 30] # For finding lexical task heads 
    task_relation_dict = {}
    task_relation_dict["python"] = ["python"]
    task_relation_dict["js"] = ["JavaScript"]

    result_dict = intervene_generation(
        model=model, 
        convert_2_task=convert_2_task, 
        original_task=original_task, 
        gen_prompt_type="problem", 
        exp_size=exp_size, 
        n_generate_tokens=n_generate_tokens, 
        intervene_positions=intervene_positions, 
        scaling_factor=scaling_factor, 
        batch_size=batch_size, 
        component_type="Relation", 
        save_root=save_root, 
        task_relation_dict=task_relation_dict,
        k_list=k_list,
    )
    
    # Save interv_generation in json file 
    save_folder = os.path.join(save_root, model_name, 
        "code_generation_intervention", )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder,
        f"steer_{original_task}_to_{convert_2_task}_sf{int(scaling_factor)}_MaxIntervPos{max_terv_pos}_ex{exp_size}_bs{batch_size}.json")
    with open(save_path,
        "w") as f:
        json.dump(result_dict, f)
