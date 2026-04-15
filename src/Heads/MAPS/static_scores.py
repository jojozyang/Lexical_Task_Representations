import os
import torch
from transformers import AutoTokenizer
import numpy as np
import transformer_lens
import sys
from datetime import datetime
torch.set_default_device("cuda")

from utils.utils import load_dataset#,get_k
from utils.maps import MAPS


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Usage: python experiment1_correlative.py <model_name>")
        sys.exit(1)

    # Variables to be set
    result_root_dir = "/oscar/data/superlab/projects/EP_IP/output/Heads/MAPS"
    #TODO: consider use the same datasets as other analysis 
    dataset_path = "/oscar/data/epavlick/zyang220/projects_2025/EP_IP/src/Heads/MAPS/datasets"
    dataset_list = [
        'country-capital', 'product-company',
    ]
    abstract_relation = False # True for Relation Propagation heads and False for Attribute Extraction heads
    k = 10
    apply_first_mlp = False # for static scores

    # Model 
    model_name = sys.argv[1]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] running static score experiment for model: {model_name}")
    if "meta-llama" in model_name:
        model_name_ = "".join(model_name.split("Meta-"))
    else:
        model_name_ = model_name
    model = transformer_lens.HookedTransformer.from_pretrained_no_processing(model_name_, device_map="auto")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    cfg = model.cfg
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Output dir 
    for abstract_relation in [False, True]: # True for Relation Propagation heads and False for Attribute Extraction heads
        model_folder_name = model_name.split('/')[-1]
        if abstract_relation:
            output_folder_name = "Relation_Propagation_heads"
        else:
            output_folder_name = "Attribute_Extraction_heads"

        # initialize MAPS and
        maps = MAPS(model, tokenizer, abstract_relation=abstract_relation)

        # run experiment
        for relation_name in dataset_list:
            dataset = load_dataset(relation_name, dataset_path)
            #k = get_k(model_name, relation_name)
            print("k is", k)
            relation_scores, suppression_relation_scores = maps.calc_relation_scores(
            dataset, relation_name, apply_first_mlp, k)
            # save npy
            if apply_first_mlp:
                mlp_application_folder_name = "w_1stmlp"
            else:
                mlp_application_folder_name = "wo_1stmlp"
            results_dir = os.path.join(result_root_dir, 
                model_folder_name, output_folder_name, 
                "static_scores", mlp_application_folder_name, f"k{k}", relation_name)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            np.save(os.path.join(results_dir, "relation_scores.npy"), relation_scores)
            np.save(os.path.join(results_dir, "suppression_relation_scores.npy"), suppression_relation_scores)
    
    print("done")