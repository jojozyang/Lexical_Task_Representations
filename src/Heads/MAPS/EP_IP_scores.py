import os
import sys
from datetime import datetime
import torch
from transformers import AutoTokenizer
import numpy as np
import transformer_lens
torch.set_default_device("cuda")
import pandas as pd
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
import torch.nn.functional as F
from scipy.stats import pearsonr

from utils.utils import load_dataset#, get_k
from utils.paper_utils import category_to_category_name, relation_to_category, format_output_df, relation_to_fixed_name
from utils.maps import MAPS


class CorrelativeExperiment:

    def __init__(self, maps: MAPS, model_name, experiment_name, 
        cfg, n_templates, results_dir, dataset_path, 
        ICL_INST, k, apply_first_mlp
    ):
        self.model_name = model_name
        self.maps = maps
        self.experiment_name = experiment_name
        self.cfg = cfg
        self.relation_scores_cache = {}
        self.n_templates = n_templates
        self.results_dir = results_dir
        self.dynamic_scores_output_dir = os.path.join(self.results_dir, "dynamic_scores")
        self.ICL_INST = ICL_INST
        self.dataset_path = dataset_path
        if not os.path.exists(self.dynamic_scores_output_dir):
            os.makedirs(self.dynamic_scores_output_dir)
        self.correlations_output_dir = os.path.join(self.results_dir, "correlations")
        if not os.path.exists(self.correlations_output_dir):
            os.makedirs(self.correlations_output_dir)
        self.k = k
        self.apply_first_mlp = apply_first_mlp

    def calc_dynamic_relation_scores_multiple_templates(self, relation_name, dataset):
        #k = get_k(self.model_name, relation_name)
        k = self.k
        print("k is", k)
        dynamic_scores_dic_multiple_prompts = {}
        for dic in [dynamic_scores_dic_multiple_prompts]:
            for key in ["w_context_dynamic_relation_scores", 
                        "w_context_suppression_dynamic_relation_scores", 
                        "wo_context_dynamic_relation_scores", 
                        "wo_context_suppression_dynamic_relation_scores"]:
                dic[key] = []
        TEMPLATES = self.load_json_template(relation_name)
        for template in TEMPLATES:
            dynamic_scores_dic = self.maps.calc_dynamic_relation_scores(dataset,relation_name,template,k)
            for (dynamic_scores_key, dynamic_scores_values) in dynamic_scores_dic.items():
                dynamic_scores_dic_multiple_prompts[dynamic_scores_key].append(dynamic_scores_values)
        for (dynamic_scores_key, dynamic_scores_lst) in dynamic_scores_dic_multiple_prompts.items():
            dynamic_scores_dic_multiple_prompts[dynamic_scores_key] = np.concatenate(dynamic_scores_lst)
        return dynamic_scores_dic_multiple_prompts

    def load_json_template(self, relation_name):
        if self.ICL_INST == "ICL":
            with open(os.path.join(self.dataset_path, relation_name, "templates_ICL.json"), "r") as f:
                template = json.load(f)
        else:
            with open(os.path.join(self.dataset_path, relation_name, "templates_INST.json"), "r") as f:
                template = json.load(f)
        return template

    def print_pvals(self, summary_df):
        max_pval = -float('inf')
        for col in summary_df.columns:
            if "pval" in col:
                max_pval = max(max_pval, summary_df[col].max())
        print(f"{self.status()} max_pval: {max_pval:.1e}", flush=True)
        pvalues_df = summary_df[["relation"]+[col for col in summary_df.columns if "pval" in col]]
        print(pvalues_df, flush=True)

    def summary_df_to_latex(self, df):
        for is_suppression in [False,True]:
            suppression_str = "supp_" if is_suppression else ""
            df_entries = []
            for _,row in df.iterrows():
                relation_name = row["relation"]
                fixed_relation_name = relation_to_fixed_name(relation_name)
                category_name = category_to_category_name[relation_to_category[relation_name]]
                relation_scores, suppression_relation_scores = self.relation_scores_cache[relation_name] 
                max_relation_score = relation_scores.max() if (not is_suppression) else suppression_relation_scores.max()
                df_entries.append({
                    "Category": category_name,
                    "Relation": fixed_relation_name,
                    "Max relation score (over heads)": f"{max_relation_score:.2f}",
                    "Correlation w/o context": f"{row[f'{suppression_str}corr_wo_context']:.2f}",
                    "Correlation w/ context": f"{row[f'{suppression_str}corr_w_context']:.2f}",
                })
            output_df = pd.DataFrame(df_entries)  
            output_df = format_output_df(output_df)
            stripped_model_name = self.model_name.split("/")[-1]
            output_path = os.path.join(self.correlations_output_dir, f"{stripped_model_name}_{suppression_str}dynamic_scores.tex")
            output_df.to_latex(output_path, index=False)
        
    def save_results(self, summary_list):
        # with open(os.path.join(self.results_dir, "templates.json"), "w") as f:
        #     f.write(json.dumps(TEMPLATES,indent=2))
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(os.path.join(self.correlations_output_dir, "summary.csv"),index=False)
        self.summary_df_to_latex(summary_df)
        self.print_pvals(summary_df)

    def save_dynamic_scores(self, dynamic_scores_dic_multiple_prompts, relation_name):
        dynamic_scores_dic = {key:val.flatten() for key,val in dynamic_scores_dic_multiple_prompts.items()}
        dynamic_scores_df = pd.DataFrame(dynamic_scores_dic)
        dir = os.path.join(self.dynamic_scores_output_dir, relation_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        dynamic_scores_df.to_csv(os.path.join(dir,"scores.csv"),index=False)

    def get_static_scores(self, relation_name, dataset):
        #apply_first_mlp = True if ("Llama-3.1-70B" not in self.model_name) else False
        apply_first_mlp = self.apply_first_mlp
        k = self.k
        print("k is", k)
        #k = get_k(self.model_name, relation_name)
        relation_scores, suppression_relation_scores = self.maps.calc_relation_scores(dataset, relation_name, apply_first_mlp, k)
        self.relation_scores_cache[relation_name] = (relation_scores, suppression_relation_scores)
        return relation_scores, suppression_relation_scores

    def calc_correlations(self, relation_name, relation_scores, suppression_relation_scores, dynamic_scores_dic_multiple_prompts):
        corr_wo_context, pval_wo = \
            pearsonr(np.concatenate([relation_scores]*self.n_templates).flatten(), dynamic_scores_dic_multiple_prompts["wo_context_dynamic_relation_scores"].flatten())
        corr_w_context, pval_w = \
            pearsonr(np.concatenate([relation_scores]*self.n_templates).flatten(), dynamic_scores_dic_multiple_prompts["w_context_dynamic_relation_scores"].flatten())
        supp_corr_wo_context, supp_pval_wo_context = \
            pearsonr(np.concatenate([suppression_relation_scores]*self.n_templates).flatten(), dynamic_scores_dic_multiple_prompts["wo_context_suppression_dynamic_relation_scores"].flatten())
        supp_corr_w_context, supp_pval_w_context = \
            pearsonr(np.concatenate([suppression_relation_scores]*self.n_templates).flatten(), dynamic_scores_dic_multiple_prompts["w_context_suppression_dynamic_relation_scores"].flatten())
        results = {
                "relation": relation_name,
                "corr_wo_context": corr_wo_context,
                "corr_w_context": corr_w_context,
                "supp_corr_wo_context": supp_corr_wo_context,
                "supp_corr_w_context": supp_corr_w_context,
                "pval_wo_context": pval_wo,
                "pval_w_context": pval_w,
                "supp_pval_wo_context": supp_pval_wo_context,
                "supp_pval_w_context": supp_pval_w_context
            }
        return results

    def run_correlative_experiment_one_relation(self, relation_name, dataset):
        print(f"{self.status()} running correlative experiment for {relation_name}")
        #relation_scores, suppression_relation_scores = self.get_static_scores(relation_name, dataset)

        dynamic_scores_dic_multiple_prompts = self.calc_dynamic_relation_scores_multiple_templates(relation_name, dataset)
        self.save_dynamic_scores(dynamic_scores_dic_multiple_prompts, relation_name)
        #NOTE: we don't calculate correlations for now
        #results = self.calc_correlations(relation_name, relation_scores, suppression_relation_scores, dynamic_scores_dic_multiple_prompts)
        #return results
    
    def run_experiment(self, dataset_list=None):
        #summary_list = []
        if dataset_list is None:
            dataset_list = os.listdir("datasets")
        for relation_name in tqdm(dataset_list):
            dataset = load_dataset(relation_name, self.dataset_path)
            if not self.maps.is_valid_dataset_for_model(dataset, relation_name):
                continue
            print(f"{self.status()} relation: {relation_name}")
            #results = self.run_correlative_experiment_one_relation(relation_name, dataset)
            self.run_correlative_experiment_one_relation(relation_name, dataset)
            #summary_list.append(results)
        #self.save_results(summary_list)
        
    def status(self):
        return f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Correlative experiment for {self.model_name}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Usage: python experiment1_correlative.py <model_name>")
        sys.exit(1)

    # Variables to be set
    k=10 # Top k tokens to consider when calculating the scores
    dataset_list = [
       # 'country-capital', 'product-company', 
        'antonym', 
    ]
    apply_first_mlp = False
    #TODO: consider use the same datasets as other analysis 
    dataset_path = "/oscar/data/epavlick/zyang220/projects_2025/EP_IP/src/Heads/MAPS/datasets"
    result_root_dir = "/oscar/data/superlab/projects/EP_IP/output/Heads/MAPS"
    n_templates = 5

    # Model 
    model_name = sys.argv[1]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] running experiment for model: {model_name}")
    #TODO: replace with: model_name = model_name.split('/')[-1]
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

    # Run experiment for both ICL & INST, as well as both Relation Propagation & Attribute Extraction heads
    for ICL_INST in ["ICL", "INST"]: # INST:(EP), ICL:(IP)
        #for abstract_relation in [False, True]: # True for Relation Propagation heads and False for Attribute Extraction heads
        for abstract_relation in [False]:
            # Result dir 
            experiment_name = "multiple_prompts"
            model_folder_name = model_name.split('/')[-1]
            if abstract_relation:
                output_folder_name = "Relation_Propagation_heads"
            else:
                output_folder_name = "Attribute_Extraction_heads"
            results_dir = os.path.join(result_root_dir, 
                model_folder_name, output_folder_name, 
                f"{ICL_INST}_scores", experiment_name, f"k{k}")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            print("results_dir", results_dir)
            
            # initialize MAPS and run experiment
            maps = MAPS(model, tokenizer, abstract_relation=abstract_relation)
            correlative_experiment = CorrelativeExperiment(maps, model_name, experiment_name, 
                cfg, n_templates, results_dir, dataset_path, 
                ICL_INST, k, apply_first_mlp)
            correlative_experiment.run_experiment(dataset_list=dataset_list)
    print("done")