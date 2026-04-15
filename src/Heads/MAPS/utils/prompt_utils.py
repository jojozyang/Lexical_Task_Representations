import os
import json
from transformer_lens import HookedTransformer
from Shared_utils.prompt_utils import create_few_shot_prompts


def generate_instruction_prompts_lens(
    model: HookedTransformer, d_name: str, instruction: str, 
    EXP_SIZE=100, n_samples = None, project_root = None,
):
    
    with open(os.path.join(project_root, "datasets", "abstractive", f"{d_name}.json")) as f: 
        dataset_raw = json.load(f)
    if n_samples is None: 
        n_samples = len(dataset_raw)
    prompts = [instruction.format(input=data['input']) for data in dataset_raw[:n_samples]]
    answers = [data['output'] for data in dataset_raw[:n_samples]]

    # Select prompt-answer pairs which model predicted correclty.
    selected_prompts = [] 
    selected_answers = [] 
    for p, a in zip(prompts, answers): 
        a_token = model.tokenizer(' ' + a, add_special_tokens=False).input_ids[0]
        logits, _ = model.run_with_cache(p, return_cache_object=False)
        pred = logits[:, -1].argmax().item()
        if a_token == pred:
            selected_prompts.append(p) 
            selected_answers.append(' ' + a)
    return selected_prompts[:EXP_SIZE], selected_answers[:EXP_SIZE]

def generate_example_prompts_lens(
    model: HookedTransformer, d_name: str, n_shot: int=5,
    EXP_SIZE=100, n_samples=300, project_root = None,
):
    
    with open(os.path.join(project_root, "datasets", "abstractive", f"{d_name}.json")) as f: 
        dataset_raw = json.load(f)
    if n_samples is None: 
        n_samples = len(dataset_raw)
    prompts, answers, _ = create_few_shot_prompts(
        dataset_raw[:n_samples], n_shot = n_shot, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    

    # Select prompt-answer pairs which model predicted correclty.
    selected_prompts = [] 
    selected_answers = [] 
    for p, a in zip(prompts, answers): 
        a_token = model.tokenizer(a, add_special_tokens=False).input_ids[0]
        logits, _ = model.run_with_cache(p, return_cache_object=False)
        pred = logits[:, -1].argmax().item()
        if a_token == pred:
            selected_prompts.append(p) 
            selected_answers.append(a)

    return selected_prompts[:EXP_SIZE], selected_answers[:EXP_SIZE]
    
