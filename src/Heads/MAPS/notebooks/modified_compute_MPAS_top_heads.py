def compute_MPAS_top_heads(result_root_path: str, model_name: str, head_type: str, 
    k: int, dataset_name: str, n_templates_INST: int, n_layers: int, n_heads: int, n_top_heads: int
):
    """Compute top heads for both ICL and INST methods.
    
    Args:
        result_root_path: Root path for results
        model_name: Name of the model
        head_type: Type of head analysis
        k: Number of few-shot examples
        dataset_name: Name of the dataset
        n_templates_INST: Number of templates for INST
        n_layers: Number of layers in the model
        n_heads: Number of heads per layer
        n_top_heads: Number of top heads to return
    
    Returns:
        tuple: (top_heads_ICL, top_heads_INST) where each is a list of (layer_idx, head_idx, score) tuples
    """
    import os
    import pandas as pd
    import numpy as np
    import torch
    
    top_heads_ICL = None
    top_heads_INST = None
    
    # Process ICL
    try:
        result_path_ICL = os.path.join(result_root_path,
            model_name, head_type, f"ICL_scores/multiple_prompts/k{k}/dynamic_scores",
            dataset_name, "scores.csv")
        scores_df_ICL = pd.read_csv(result_path_ICL)
        scores_np_ICL = scores_df_ICL["w_context_dynamic_relation_scores"].to_numpy(dtype=np.float32)
        scores_reshaped_ICL = scores_np_ICL.reshape((n_templates_INST, n_layers, n_heads))

        # Calculate the mean across the "templates" axis (axis=0)
        mean_score_np_ICL = scores_reshaped_ICL.mean(axis=0)
        score_tensor_ICL = torch.from_numpy(mean_score_np_ICL).float()

        # Compute Top Heads (L,H) for ICL
        flattened_score_ICL = score_tensor_ICL.view(-1) # Flatten the tensor for topk
        topk_vals_ICL, topk_inds_ICL = torch.topk(flattened_score_ICL, k=n_top_heads, largest=True) # Perform topk
        layer_indices_ICL, head_indices_ICL = torch.unravel_index(topk_inds_ICL, score_tensor_ICL.shape)
        top_lh_temp_ICL = torch.stack((layer_indices_ICL, head_indices_ICL), dim=1)

        # Combine the indices and values for ICL
        top_heads_ICL = []
        for i in range(n_top_heads):
            l_idx = top_lh_temp_ICL[i, 0].item()
            h_idx = top_lh_temp_ICL[i, 1].item()
            val = round(topk_vals_ICL[i].item(), 4)
            top_heads_ICL.append((l_idx, h_idx, val))
    except Exception as e:
        print(f"Error processing ICL data: {e}")
        top_heads_ICL = None
    
    # Process INST
    try:
        result_path_INST = os.path.join(result_root_path,
            model_name, head_type, f"INST_scores/multiple_prompts/k{k}/dynamic_scores",
            dataset_name, "scores.csv")
        scores_df_INST = pd.read_csv(result_path_INST)
        scores_np_INST = scores_df_INST["w_context_dynamic_relation_scores"].to_numpy(dtype=np.float32)
        scores_reshaped_INST = scores_np_INST.reshape((n_templates_INST, n_layers, n_heads))

        # Calculate the mean across the "templates" axis (axis=0)
        mean_score_np_INST = scores_reshaped_INST.mean(axis=0)
        score_tensor_INST = torch.from_numpy(mean_score_np_INST).float()

        # Compute Top Heads (L,H) for INST
        flattened_score_INST = score_tensor_INST.view(-1) # Flatten the tensor for topk
        topk_vals_INST, topk_inds_INST = torch.topk(flattened_score_INST, k=n_top_heads, largest=True) # Perform topk
        layer_indices_INST, head_indices_INST = torch.unravel_index(topk_inds_INST, score_tensor_INST.shape)
        top_lh_temp_INST = torch.stack((layer_indices_INST, head_indices_INST), dim=1)

        # Combine the indices and values for INST
        top_heads_INST = []
        for i in range(n_top_heads):
            l_idx = top_lh_temp_INST[i, 0].item()
            h_idx = top_lh_temp_INST[i, 1].item()
            val = round(topk_vals_INST[i].item(), 4)
            top_heads_INST.append((l_idx, h_idx, val))
    except Exception as e:
        print(f"Error processing INST data: {e}")
        top_heads_INST = None
    
    return top_heads_ICL, top_heads_INST 