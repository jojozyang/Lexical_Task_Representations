# import torch
# from nnsight import LanguageModel
# from Shared_utils.wrapper import get_accessor_config, get_model_specs, ModelAccessor
# from Shared_utils.shared_utils import *

# def set_activation_rank_logit(
#     model: LanguageModel, batch_size: int = 8,
#     clean_prompts: list[str] = None, corrupt_prompts: list[str] = None, answers: list[str] = None,
#     grouped_heads: dict[int, list[int]] = None, 
#     all_clean_tokens: torch.Tensor = None,
#     all_corrupt_tokens: torch.Tensor = None,
#     all_answer_tokens: torch.Tensor = None,
#     remote: bool = False, prepend_bos:int = True,
#     scaling_factor: float = 1,
# ) -> torch.Tensor:
#     """ 

#     Args:
#         model: Language model to analyze
#         clean_prompts: List of prompts
#         corrupt_prompts: List of prompts
#         answers: List of expected answer strings
#         all_clean_tokens: Pre-tokenized clean tokens
#         all_corrupt_tokens: Pre-tokenized corrupt tokens
#         all_answer_tokens: Pre-tokenized answer tokens
#         grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. 
#         batch_size: Size of each processing batch
#         remote: Whether to run model remotely
       
#     Returns:
#         intervention_ranks (torch.Tensor): ranks of the intervention tokens
#     """
#     # Get model specs
#     spec = get_model_specs(model)
#     n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
#     d_head = d_model // n_heads

#     # Tokenize all data upfront
#     if all_clean_tokens is None:
#         all_clean_tokens = model.tokenizer(
#             clean_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_corrupt_tokens is None:
#         all_corrupt_tokens = model.tokenizer(
#             corrupt_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_answer_tokens is None:
#         all_answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

#     # Input validation
#     total_samples = all_clean_tokens.shape[0]
#     if not (total_samples == all_corrupt_tokens.shape[0] == len(all_answer_tokens)):
#         raise ValueError("All input lists must have the same length")

#     # Initialize results storage 
#     intervention_ranks = []
#     # corrupt_ranks = []
#     # clean_ranks = []
#     logit_recovery = 0

#     # Process in batches
#     # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
#     for batch_start in range(0, total_samples, batch_size):
#         batch_end = min(batch_start + batch_size, total_samples)
#         current_batch_size = batch_end - batch_start

#         # Get current batch tokens from pre-tokenized data
#         batch_clean_tokens = all_clean_tokens[batch_start:batch_end]
#         batch_corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
#         batch_answer_tokens = all_answer_tokens[batch_start:batch_end]

#         # Get clean and corrupt logits at answer idx
#         clean_logits = model.trace(batch_clean_tokens, trace=False, remote=remote)["logits"][:, -1]
#         corrupt_logits = model.trace(batch_corrupt_tokens, trace=False, remote=remote)["logits"][:, -1]
#         clean_answer_logits = clean_logits[torch.arange(current_batch_size), batch_answer_tokens]
#         corrupt_answer_logits = corrupt_logits[torch.arange(current_batch_size), batch_answer_tokens]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)

#         # Collect clean activations of grouped heads
#         clean_act = {}
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_clean_tokens):
#                 for layer in grouped_heads.keys(): # only need to cache relevant layers' activation 
#                     attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)

#                     for head in grouped_heads[layer]: # only need to cache relevant heads' activation 
#                         clean_act[(layer, head)] = attn_reshaped[:, head, :].save()
#                 #clean_logits = model.output.logits[:,-1].save()
                    
#         # Corrupt runs, patch the clean activations of grouped heads
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_corrupt_tokens):
#                 for layer in grouped_heads.keys():
#                     interv_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     interv_reshaped = interv_attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in grouped_heads[layer]: 
#                         # patch clean activation of grouped heads
#                         interv_reshaped[:, head, :] = scaling_factor * clean_act[(layer, head)] 
                
#             # Get intervention logits
#             interv_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()
#             interv_logits = model.output.logits[:,-1].save() 

#         # Calculate normalized intervention effect on logits
#         intervention_diff = interv_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         batch_effect = intervention_diff.mean() / baseline.mean()
#         logit_recovery += batch_effect * (current_batch_size / total_samples) # Accumulate batch results (weighted by batch size)
        
#         #batch_clean_ranks, _ = compute_token_rank_prob(clean_logits, batch_answer_tokens)
#         batch_intervention_ranks, _ = compute_token_rank_prob(interv_logits, batch_answer_tokens) 
#         #batch_corrupt_ranks, _ = compute_token_rank_prob(corrupt_logits, batch_answer_tokens) 

#         del interv_logits, corrupt_logits, clean_logits, clean_act # free memory
#         free_unused_cuda_memory()       
      
#         intervention_ranks.append(batch_intervention_ranks.to('cpu'))
#         #corrupt_ranks.append(batch_corrupt_ranks.to('cpu'))
#         #clean_ranks.append(batch_clean_ranks.to('cpu'))
    
#     intervention_ranks = torch.cat(intervention_ranks)
   
#     return intervention_ranks, logit_recovery

# def set_activation_rank_logit_average(
#     model: LanguageModel, batch_size: int = 8,
#     grouped_components: dict[int, list[int]] = None, 
#     all_clean_tokens: torch.Tensor = None,
#     all_corrupt_tokens: torch.Tensor = None,
#     all_answer_tokens: torch.Tensor = None,
#     remote: bool = False, heads_or_neurons:str="heads",
#     scaling_factor: float = 1,
# ) -> torch.Tensor:
#     """ 
#     Patch corrupt with the average of clean activation 
#     Args:
#         model: Language model to analyze
#         clean_prompts: List of prompts
#         corrupt_prompts: List of prompts
#         answers: List of expected answer strings
#         all_clean_tokens: Pre-tokenized clean tokens
#         all_corrupt_tokens: Pre-tokenized corrupt tokens
#         all_answer_tokens: Pre-tokenized answer tokens
#         grouped_components: a dict where a list of component_index of a particular layer are grouped under its layer key. 
#         batch_size: Size of each processing batch
#         remote: Whether to run model remotely
       
#     Returns:
#         intervention_ranks (torch.Tensor): ranks of the intervention tokens
#         logit_recovery (float): logit recovery of the intervention
#     """
#     # Input validation
#     # all_answer_tokens.shape[0] == all_corrupt_tokens.shape[0] 

#     # Get model specs
#     spec = get_model_specs(model)
#     n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
#     d_head = d_model // n_heads
#     d_mlp = spec["d_mlp"]

#     clean_sample_size = all_clean_tokens.shape[0]
#     corrupt_sample_size = all_corrupt_tokens.shape[0]

#     # Clean runs 
#     if heads_or_neurons == "heads":
#         clean_act = torch.empty(clean_sample_size, n_layers, n_heads, d_head)
#     elif heads_or_neurons == "neurons":
#         clean_act = torch.empty(clean_sample_size, n_layers, d_mlp)
#     else:
#         raise ValueError(f"heads_or_neurons {heads_or_neurons} not supported")
#     for batch_start in range(0, clean_sample_size, batch_size):
#         batch_end = min(batch_start + batch_size, clean_sample_size)
#         current_batch_size = batch_end - batch_start

#         # Get current batch tokens from pre-tokenized data
#         batch_clean_tokens = all_clean_tokens[batch_start:batch_end]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)

#         # Collect clean activations of grouped heads
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_clean_tokens):
#                 for layer in range(n_layers):
#                     if heads_or_neurons == "heads":
#                         attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                         attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
#                         clean_act[batch_start:batch_end, layer, :, :] = attn_reshaped.save()
#                     elif heads_or_neurons == "neurons":
#                         mlp_output = accessor.layers[layer].mlp.down_proj.unwrap().input[:, -1]
#                         clean_act[batch_start:batch_end, layer, :] = mlp_output.save()
#                     else:
#                         raise ValueError(f"heads_or_neurons {heads_or_neurons} not supported")
#     if heads_or_neurons == "heads":
#         assert clean_act.shape == (clean_sample_size, n_layers, n_heads, d_head)
#     elif heads_or_neurons == "neurons":
#         assert clean_act.shape == (clean_sample_size, n_layers, d_mlp)
#     else:
#         raise ValueError(f"heads_or_neurons {heads_or_neurons} not supported")
    
#     clean_act_mean = clean_act.mean(dim=0) # (n_layers, n_heads, d_head) or (n_layers, d_mlp)
    
#     del clean_act # free memory
#     free_unused_cuda_memory()

#     # Initialize results storage 
#     intervention_ranks = []
#     #logit_recovery = 0
#     # Corrupt runs, patch the clean activations of grouped heads
#     for batch_start in range(0, corrupt_sample_size, batch_size):
#         batch_end = min(batch_start + batch_size, corrupt_sample_size)
#         current_batch_size = batch_end - batch_start 

#         # Get current batch tokens from pre-tokenized data
#         batch_clean_tokens = all_clean_tokens[batch_start:batch_end]
#         batch_corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
#         batch_answer_tokens = all_answer_tokens[batch_start:batch_end]

#         # Get clean and corrupt logits at answer idx
#         # clean_logits = model.trace(batch_clean_tokens, trace=False, remote=remote)["logits"][:, -1]
#         # corrupt_logits = model.trace(batch_corrupt_tokens, trace=False, remote=remote)["logits"][:, -1]
#         # clean_answer_logits = clean_logits[torch.arange(current_batch_size), batch_answer_tokens]
#         # corrupt_answer_logits = corrupt_logits[torch.arange(current_batch_size), batch_answer_tokens]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_corrupt_tokens):
#                 for layer in grouped_components.keys():
#                     component_index_list = grouped_components[layer]
#                     # patch clean activation of grouped components
#                     if heads_or_neurons == "heads":
#                         interv_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                         interv_reshaped = interv_attn_output.reshape(current_batch_size, n_heads, d_head)
#                         interv_reshaped[:, component_index_list, :] = scaling_factor * clean_act_mean[layer, component_index_list, :] 
#                     elif heads_or_neurons == "neurons":
#                         interv_mlp_output = accessor.layers[layer].mlp.down_proj.unwrap().input[:, -1]
#                         interv_mlp_output[:, component_index_list] = scaling_factor * clean_act_mean[layer, component_index_list] 
#                     else:
#                         raise ValueError(f"heads_or_neurons {heads_or_neurons} not supported")
                
#             # Get intervention logits
#             #interv_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()
#             interv_logits = model.output.logits[:,-1].save() 

#         # Calculate normalized intervention effect on logits
#         # intervention_diff = interv_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         # baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         # batch_effect = intervention_diff.mean() / baseline.mean()
#         # logit_recovery += batch_effect * (current_batch_size / corrupt_sample_size) # Accumulate batch results (weighted by batch size)
        
#         #batch_clean_ranks, _ = compute_token_rank_prob(clean_logits, batch_answer_tokens)
#         batch_intervention_ranks, _ = compute_token_rank_prob(interv_logits, batch_answer_tokens) 
#         #batch_corrupt_ranks, _ = compute_token_rank_prob(corrupt_logits, batch_answer_tokens) 

#         del interv_logits
#         #del interv_logits, corrupt_logits, clean_logits # free memory
#         free_unused_cuda_memory()       
      
#         intervention_ranks.append(batch_intervention_ranks.to('cpu'))
#         #corrupt_ranks.append(batch_corrupt_ranks.to('cpu'))
#         #clean_ranks.append(batch_clean_ranks.to('cpu'))
    
#     intervention_ranks = torch.cat(intervention_ranks)
   
#     return intervention_ranks


# def set_ablation_rank_logit(
#     model: LanguageModel, batch_size: int = 8,
#     clean_prompts: list[str] = None, corrupt_prompts: list[str] = None, answers: list[str] = None,
#     grouped_heads: dict[int, list[int]] = None, 
#     remote: bool = False, prepend_bos:int = True,
#     scaling_factor: float = 1.0,
#     all_clean_tokens: torch.Tensor = None,
#     all_corrupt_tokens: torch.Tensor = None,
#     all_answer_tokens: torch.Tensor = None,
# ) -> torch.Tensor:
#     """ 

#     Args:
#         model: Language model to analyze
#         clean_prompts: List of prompts
#         corrupt_prompts: List of prompts
#         answers: List of expected answer strings
#         grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. 
#         all_clean_tokens: Pre-tokenized clean tokens
#         all_corrupt_tokens: Pre-tokenized corrupt tokens
#         all_answer_tokens: Pre-tokenized answer tokens
#         batch_size: Size of each processing batch
#         remote: Whether to run model remotely
       
#     Returns:
#         intervention_ranks (torch.Tensor): ranks of the intervention tokens
#         clean_ranks (torch.Tensor): ranks of the clean tokens
#         rank_demotion (float): rank demotion of the intervention tokens
#         logit_degradation (float): logit degradation of the intervention tokens
#     """
    

#     # Get model specs
#     spec = get_model_specs(model)
#     n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
#     d_head = d_model // n_heads

#     # Tokenize all data upfront
#     if all_clean_tokens is None:
#         all_clean_tokens = model.tokenizer(
#             clean_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_corrupt_tokens is None:
#         all_corrupt_tokens = model.tokenizer(
#             corrupt_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_answer_tokens is None:
#         all_answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

#     # Input validation
#     total_samples = all_clean_tokens.shape[0]
#     if not (total_samples == all_corrupt_tokens.shape[0] == len(all_answer_tokens)):
#         raise ValueError("All input lists must have the same length")

#     # Initialize results storage 
#     intervention_ranks = []
#     # corrupt_ranks = []
#     # clean_ranks = []
#     logit_degradation = 0

#     # Process in batches
#     # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
#     for batch_start in range(0, total_samples, batch_size):
#         batch_end = min(batch_start + batch_size, total_samples)
#         current_batch_size = batch_end - batch_start

#         # Get current batch tokens from pre-tokenized data
#         batch_clean_tokens = all_clean_tokens[batch_start:batch_end]
#         batch_corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
#         batch_answer_tokens = all_answer_tokens[batch_start:batch_end]

#         # Get clean and corrupt logits at answer idx
#         clean_logits = model.trace(batch_clean_tokens, trace=False, remote=remote)["logits"][:, -1]
#         corrupt_logits = model.trace(batch_corrupt_tokens, trace=False, remote=remote)["logits"][:, -1]
#         clean_answer_logits = clean_logits[torch.arange(current_batch_size), batch_answer_tokens]
#         corrupt_answer_logits = corrupt_logits[torch.arange(current_batch_size), batch_answer_tokens]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)

#         # Collect corrupt activations of grouped heads
#         corrupt_act = {}
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_corrupt_tokens):
#                 for layer in grouped_heads.keys(): # only need to cache relevant layers' activation 
#                     attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)

#                     for head in grouped_heads[layer]: # only need to cache relevant heads' activation 
#                         corrupt_act[(layer, head)] = attn_reshaped[:, head, :].save()
#                 #clean_logits = model.output.logits[:,-1].save()
                    
#         # Clean runs, patch the corrupt activations of grouped heads
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_clean_tokens):
#                 for layer in grouped_heads.keys():
#                     interv_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     interv_reshaped = interv_attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in grouped_heads[layer]: 
#                         # patch clean activation of grouped heads
#                         interv_reshaped[:, head, :] = corrupt_act[(layer, head)] 
                
#             # Get intervention logits
#             interv_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()
#             interv_logits = model.output.logits[:,-1].save() 

#         # Calculate normalized intervention effect on logits
#         intervention_diff = clean_answer_logits.to("cpu") - interv_answer_logits.to("cpu")
#         baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         batch_effect = intervention_diff.mean() / baseline.mean()
#         logit_degradation += batch_effect * (current_batch_size / total_samples) # Accumulate batch results (weighted by batch size)
        
#         #batch_clean_ranks, _ = compute_token_rank_prob(clean_logits, batch_answer_tokens)
#         batch_intervention_ranks, _ = compute_token_rank_prob(interv_logits, batch_answer_tokens) 
#         #batch_corrupt_ranks, _ = compute_token_rank_prob(corrupt_logits, batch_answer_tokens) 

#         del interv_logits, corrupt_logits, clean_logits, corrupt_act # free memory
#         free_unused_cuda_memory()       
      
#         intervention_ranks.append(batch_intervention_ranks.to('cpu'))
#         # corrupt_ranks.append(batch_corrupt_ranks.to('cpu'))
#         # clean_ranks.append(batch_clean_ranks.to('cpu'))
    
#     intervention_ranks = torch.cat(intervention_ranks)
#     #corrupt_ranks = torch.cat(corrupt_ranks)
#     #clean_ranks = torch.cat(clean_ranks)
   
#     return intervention_ranks, logit_degradation

# def set_component_to_logits(
#     model: LanguageModel,
#     clean_prompts: list[str] = None,
#     corrupt_prompts: list[str] = None,
#     answers: list[str] = None,
#     all_clean_tokens: torch.Tensor = None,
#     all_corrupt_tokens: torch.Tensor = None,
#     all_answer_tokens: torch.Tensor = None,
#     grouped_heads: dict[int, list[int]] = None, 
#     batch_size: int = 8,
#     remote: bool = False,
#     prepend_bos:int = True,
# ) -> torch.Tensor:
#     """ 
#     "Path patching" a set of attention heads 
#     Only activate a set of heads with clean activations, patch corrupted activations to the rest of the heads and MLPs 
    

#     Args:
#         #NOTE: Either take tokenized or string prompts/answers
#         grouped_heads: a dict where all heads of a particular layer are grouped under its layer key 

#     Returns:
#         intervention_ranks (torch.Tensor): ranks of the intervention tokens
#         logit_recovery (float)
#     """
#     # Get model specs
#     spec = get_model_specs(model)
#     n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
#     d_head = d_model // n_heads

#     # Tokenize all data upfront if not tokenized before 
#     if all_clean_tokens is None:
#         all_clean_tokens = model.tokenizer(
#             clean_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_corrupt_tokens is None:
#         all_corrupt_tokens = model.tokenizer(
#             corrupt_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_answer_tokens is None:
#         all_answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

#     # Input validation for pairwise intervention 
#     total_samples = all_clean_tokens.shape[0]
#     if not (total_samples == all_corrupt_tokens.shape[0] == len(all_answer_tokens)):
#         raise ValueError("All input (clean_prompts, corrupt_prompts, answers) must have the same length")
    
#     # Initialize results storage 
#     intervention_ranks = []
#     logit_recovery = 0
    
#     # Process in batches
#     # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
#     for batch_start in range(0, total_samples, batch_size):
#         batch_end = min(batch_start + batch_size, total_samples)
#         current_batch_size = batch_end - batch_start

#         # Store clean and corrupt activations
#         clean_attn_out = []
#         clean_mlp = []
#         corrupt_attn_out = []
#         corrupt_mlp = []

#         # Get current batch tokens from pre-tokenized data
#         batch_clean_tokens = all_clean_tokens[batch_start:batch_end]
#         batch_corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
#         batch_answer_tokens = all_answer_tokens[batch_start:batch_end]

#         # Get clean and corrupt logits at answer idx
#         clean_logits = model.trace(batch_clean_tokens, trace=False, remote=remote)["logits"][:, -1]
#         corrupt_logits = model.trace(batch_corrupt_tokens, trace=False, remote=remote)["logits"][:, -1]
#         clean_answer_logits = clean_logits[torch.arange(current_batch_size), batch_answer_tokens]
#         corrupt_answer_logits = corrupt_logits[torch.arange(current_batch_size), batch_answer_tokens]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)

#         # Collect clean activations
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_clean_tokens):
#                 for layer in range(n_layers):
#                     attn_out = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_out_reshaped = attn_out.reshape(current_batch_size, n_heads, d_head).save()
#                     clean_attn_out.append(attn_out_reshaped)

#                     mlp_out = accessor.layers[layer].mlp.unwrap().output[:, -1].save()
#                     clean_mlp.append(mlp_out)
    
#         # Collect corrupt activations
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_corrupt_tokens):
#                 for layer in range(n_layers):
#                     attn_out = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_out_reshaped = attn_out.reshape(current_batch_size, n_heads, d_head).save()
#                     corrupt_attn_out.append(attn_out_reshaped)

#                     mlp_out = accessor.layers[layer].mlp.unwrap().output[:, -1].save()
#                     corrupt_mlp.append(mlp_out)

#         # Patch clean activations to selected heads, patch corrupted activations to the rest of the heads and MLPs 
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_corrupt_tokens):
#                 # Freeze all heads to corrupt values
#                 for freeze_layer in range(n_layers):
#                     freeze_attn_out = accessor.layers[freeze_layer].attention.output.unwrap().input[:, -1]
#                     freeze_attn_out_reshaped = freeze_attn_out.reshape(current_batch_size, n_heads, d_head)
#                     freeze_attn_out_reshaped[...] = corrupt_attn_out[freeze_layer]

#                     # Patch selected heads with clean activation
#                     if freeze_layer in grouped_heads.keys():
#                         head_list = grouped_heads[freeze_layer]
#                         freeze_attn_out_reshaped[:, head_list, :] = clean_attn_out[freeze_layer][:, head_list, :]

#                     # Freeze MLPs to corrupt values
#                     freeze_mlp = accessor.layers[freeze_layer].mlp.unwrap().output[:, -1]
#                     freeze_mlp[...] = corrupt_mlp[freeze_layer]

#             # Get intervention logits
#             interv_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()
#             interv_logits = model.output.logits[:,-1].save() 

#         # Calculate normalized intervention effect on logits
#         intervention_diff = interv_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         batch_effect = intervention_diff.mean() / baseline.mean()
#         logit_recovery += batch_effect * (current_batch_size / total_samples) # Accumulate batch results (weighted by batch size)

#         #batch_base_ranks, batch_base_probs = compute_token_rank_prob(clean_logits, batch_answer_tokens)
#         batch_intervention_ranks, _ = compute_token_rank_prob(interv_logits, batch_answer_tokens) 
#         #batch_corrupt_ranks, _ = compute_token_rank_prob(corrupt_logits, batch_answer_tokens) 
#         batch_clean_ranks, _ = compute_token_rank_prob(clean_logits, batch_answer_tokens) 

#         del interv_logits, clean_attn_out, clean_mlp, corrupt_attn_out, corrupt_mlp # free memory
#         free_unused_cuda_memory()       
      
#         intervention_ranks.append(batch_intervention_ranks.to('cpu'))
#         #corrupt_ranks.append(batch_corrupt_ranks.to('cpu'))
#         #clean_ranks.append(batch_clean_ranks.to('cpu'))
    
#     intervention_ranks = torch.cat(intervention_ranks)
#     #corrupt_ranks = torch.cat(corrupt_ranks)
#     #clean_ranks = torch.cat(clean_ranks)

#     # Calculate normalized intervention effect on ranks
#     # intervention_rank_diff = corrupt_ranks - intervention_ranks
#     # baseline_rank_diff = corrupt_ranks - clean_ranks
#     # rank_promotion = intervention_rank_diff.float().mean() / baseline_rank_diff.float().mean()

#     return intervention_ranks, logit_recovery

# def set_component_to_logits_neuron(
#     model: LanguageModel,
#     clean_prompts: list[str] = None,
#     corrupt_prompts: list[str] = None,
#     answers: list[str] = None,
#     top_neurons: dict[int, list[list]] = None,  # TODO, dict of list 
#     index_list: list[int] = None,
#     n_top_neurons: int = 1,
#     batch_size: int = 8,
#     remote: bool = False,
#     prepend_bos:int = True,
# ) -> torch.Tensor:
#     """ 
#     Only activate a set of heads, patch corrupted activations to the rest of the heads and MLPs 

#     Args:
#         model: Language model to analyze
#         clean_prompts: List of prompts
#         corrupt_prompts: List of prompts
#         answers: List of expected answer strings
#         top_neurons: a dict of list, keyed by the original index of the prompt, value is a list of ranked neurons 
        
#         batch_size: Size of each processing batch
#         remote: Whether to run model remotely

#     Returns:
#         intervention_ranks (torch.Tensor): ranks of the intervention tokens
#     """
#     # Get model specs
#     spec = get_model_specs(model)
#     n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
#     d_head = d_model // n_heads
#     d_mlp = spec["d_mlp"]

#     # Tokenize all data upfront
#     if all_clean_tokens is None:
#         all_clean_tokens = model.tokenizer(
#             clean_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_corrupt_tokens is None:
#         all_corrupt_tokens = model.tokenizer(
#             corrupt_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_answer_tokens is None:
#         all_answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

#      # Input validation
#     total_samples = all_clean_tokens.shape[0]
#     if not (total_samples == len(corrupt_prompts) == len(answers)):
#         raise ValueError("All input lists must have the same length")

#     # Store mlp and attn out across all batches
#     clean_mlp_all = []
#     corrupt_mlp_all = []
#     corrupt_attn_out_all = []

#     # Process in batches
#     # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
#     for batch_start in range(0, total_samples, batch_size):
#         batch_end = min(batch_start + batch_size, total_samples)
#         current_batch_size = batch_end - batch_start

#         # Store clean and corrupt activations
#         #clean_attn_out = []
#         clean_mlp = torch.zeros(current_batch_size, n_layers, d_mlp)
#         corrupt_attn_out = torch.zeros(current_batch_size, n_layers, n_heads, d_head)
#         corrupt_mlp = torch.zeros(current_batch_size, n_layers, d_mlp)

#         # Get current batch tokens from pre-tokenized data
#         batch_clean_tokens = all_clean_tokens[batch_start:batch_end]
#         batch_corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
#         batch_answer_tokens = all_answer_tokens[batch_start:batch_end]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)

#         # Collect clean activations
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_clean_tokens):
#                 for layer in range(n_layers):
#                     # attn_out = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     # attn_out_reshaped = attn_out.reshape(current_batch_size, n_heads, d_head).save()
#                     # clean_attn_out.append(attn_out_reshaped)
                    
#                     mlp_out = accessor.layers[layer].mlp.down_proj.unwrap().input[:, -1].save() # (batch size, d_model)
#                     clean_mlp[:, layer] = mlp_out

#         # Collect corrupt activations
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_corrupt_tokens):
#                 for layer in range(n_layers):
#                     attn_out = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_out_reshaped = attn_out.reshape(current_batch_size, n_heads, d_head).save()
#                     corrupt_attn_out[:, layer] = attn_out_reshaped

#                     mlp_in = accessor.layers[layer].mlp.down_proj.unwrap().input[:, -1].save() # (batch size, d_model)
#                     corrupt_mlp[:, layer] = mlp_in
        
#         clean_mlp_all.append(clean_mlp)
#         corrupt_mlp_all.append(corrupt_mlp)
#         corrupt_attn_out_all.append(corrupt_attn_out)

#     clean_mlp_all = torch.cat(clean_mlp_all) # (total_samples, n_layers, d_mlp)
#     corrupt_mlp_all = torch.cat(corrupt_mlp_all) # (total_samples, n_layers, d_mlp)
#     corrupt_attn_out_all = torch.cat(corrupt_attn_out_all) # (total_samples, n_layers, n_heads, d_head)

#     # Initialize results storage 
#     intervention_ranks = []
#     effects = []
#     logit_recovery = 0
#     for i in range(len(all_corrupt_tokens)):
#         corrupt_token = all_corrupt_tokens[i]
#         answer_token = [all_answer_tokens[i]]
#         clean_token = all_clean_tokens[i]
#         ori_index = index_list[i]
#         top_n_neurons = top_neurons[str(ori_index)][:n_top_neurons]
#         grouped_neurons = group_ranked_locations_by_layer(top_n_neurons)

#         # Get clean and corrupt logits at answer idx
#         clean_logit = model.trace(clean_token, trace=False, remote=remote)["logits"][:, -1]
#         corrupt_logit = model.trace(corrupt_token, trace=False, remote=remote)["logits"][:, -1]
#         clean_answer_logit = clean_logit[torch.arange(1), answer_token]
#         corrupt_answer_logit = corrupt_logit[torch.arange(1), answer_token]

#         # Patch clean activations to selected heads, patch corrupted activations to the rest of the heads and MLPs 
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(corrupt_token):
#                 # Freeze all heads & MLPs to corrupt values
#                 for freeze_layer in range(n_layers):
#                     freeze_attn_out = accessor.layers[freeze_layer].attention.output.unwrap().input[:, -1]
#                     freeze_attn_out_reshaped = freeze_attn_out.reshape(1, n_heads, d_head)
#                     freeze_attn_out_reshaped[0] = corrupt_attn_out_all[i, freeze_layer]

#                     freeze_mlp = accessor.layers[freeze_layer].mlp.down_proj.unwrap().input[:, -1]
#                     freeze_mlp[0] = corrupt_mlp_all[i, freeze_layer]

#                     # Patch selected neurons with clean activation
#                     if freeze_layer in grouped_neurons.keys():
#                         neuron_list = grouped_neurons[freeze_layer]
#                         patch_mlp = accessor.layers[freeze_layer].mlp.down_proj.unwrap().input[:, -1]
#                         patch_mlp[0, neuron_list] = clean_mlp_all[i, freeze_layer, neuron_list]
                           
#                 # Get intervention logits
#                 interv_answer_logit = accessor.lm_head.unwrap().output[:, -1][torch.arange(1), answer_token].save()
#                 interv_logit = model.output.logits[:,-1].save() 

#         # Calculate normalized intervention effect on logits
#         intervention_diff = interv_answer_logit.to("cpu") - corrupt_answer_logit.to("cpu")
#         baseline = clean_answer_logit.to("cpu") - corrupt_answer_logit.to("cpu")
#         effect = intervention_diff / baseline 
#         effects.append(effect)

#         intervention_rank, _ = compute_token_rank_prob(interv_logit, answer_token) 

#         del (interv_logit, interv_answer_logit,
#              clean_logit, corrupt_logit, clean_answer_logit, corrupt_answer_logit) # free memory
#         free_unused_cuda_memory()       
      
#         intervention_ranks.append(intervention_rank.to('cpu'))
    
#     del clean_mlp, corrupt_attn_out, corrupt_mlp
#     free_unused_cuda_memory()

#     intervention_ranks = torch.cat(intervention_ranks)

#     effects = torch.cat(effects)
#     logit_recovery = effects.mean()

#     return intervention_ranks, logit_recovery

# def eval_interface_sufficiency_rank_logit(
#     model: LanguageModel,
#     clean_prompts: list[str] = None,
#     corrupt_prompts: list[str] = None,
#     answers: list[str] = None,
#     interface_grouped_heads: dict[int, list[int]] = None, 
#     execution_grouped_heads: dict[int, list[int]] = None, 
#     batch_size: int = 8,
#     remote: bool = False,
#     prepend_bos:int = True,
# ) -> torch.Tensor:
#     """ 
#     clean run: clean prompts 
#         cache clean activations of interface heads (clean_act_interface)
#     patching run 1: corrupted prompts
#         patch in the clean activations of interface heads, cache activations of execution heads (act_execution)
#     patching run 2: corrupted prompts 
#         patch in the cached activations of execution heads  

#     Args:
#         model: Language model to analyze
#         clean_prompts: List of prompts
#         corrupt_prompts: List of prompts
#         answers: List of expected answer strings
#         interface_grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. 
#         execution_grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. 
#         batch_size: Size of each processing batch
#         remote: Whether to run model remotely

#     Returns:
#         intervention_ranks (torch.Tensor): ranks of the intervention tokens
#     """
#     # Input validation
#     total_samples = len(clean_prompts)
#     if not (total_samples == len(corrupt_prompts) == len(answers)):
#         raise ValueError("All input lists must have the same length")

#     # Get model specs
#     spec = get_model_specs(model)
#     n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
#     d_head = d_model // n_heads

#     # Tokenize all data upfront
#     #print("Tokenizing all data...")
#     all_clean_tokens = model.tokenizer(
#         clean_prompts,
#         padding=True,
#         padding_side="left",
#         add_special_tokens=prepend_bos,
#         return_tensors="pt",
#     ).input_ids

#     all_corrupt_tokens = model.tokenizer(
#         corrupt_prompts,
#         padding=True,
#         padding_side="left",
#         add_special_tokens=prepend_bos,
#         return_tensors="pt",
#     ).input_ids

#     # Get token IDs for answers
#     answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

#     # Initialize results storage 
#     intervention_ranks = []
#     corrupt_ranks = []
#     logit_recovery = 0

#     # Process in batches
#     # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
#     for batch_start in range(0, total_samples, batch_size):
#         batch_end = min(batch_start + batch_size, total_samples)
#         current_batch_size = batch_end - batch_start

#         # Get current batch tokens from pre-tokenized data
#         clean_tokens = all_clean_tokens[batch_start:batch_end]
#         corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
#         batch_answer_tokens = answer_tokens[batch_start:batch_end]

#         # Get clean and corrupt logits at answer idx
#         clean_logits = model.trace(clean_tokens, trace=False, remote=remote)["logits"][:, -1]
#         corrupt_logits = model.trace(corrupt_tokens, trace=False, remote=remote)["logits"][:, -1]
#         clean_answer_logits = clean_logits[torch.arange(current_batch_size), batch_answer_tokens]
#         corrupt_answer_logits = corrupt_logits[torch.arange(current_batch_size), batch_answer_tokens]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)

#         # Collect clean activations of interface heads
#         clean_act_interface = {}
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(clean_tokens):
#                 for layer in interface_grouped_heads.keys(): # only need to cache relevant layers' activation 
#                     attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)

#                     for head in interface_grouped_heads[layer]: # only need to cache relevant heads' activation 
#                         clean_act_interface[(layer, head)] = attn_reshaped[:, head, :].save()

#         # Corrupted runs, patch in the clean activations of interface heads and cache activations of execution heads 
#         act_execution = {}
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(corrupt_tokens):
#                 for layer in interface_grouped_heads.keys():
#                     attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in interface_grouped_heads[layer]: 
#                         # patch corrupted activations of interface heads
#                         attn_reshaped[:, head, :] = clean_act_interface[(layer, head)] 
#                 for layer in execution_grouped_heads.keys():
#                     exe_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     exe_reshaped = exe_attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in execution_grouped_heads[layer]: 
#                         # cache the activations of execution heads
#                         act_execution[(layer, head)] = exe_reshaped[:, head, :].save()
                    
#         # Corrupted runs, patch in the cached activations of execution heads 
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(corrupt_tokens):
#                 for layer in execution_grouped_heads.keys():
#                     interv_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     interv_reshaped = interv_attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in execution_grouped_heads[layer]: 
#                         # patch corrupted activation of execution heads
#                         interv_reshaped[:, head, :] = act_execution[(layer, head)] 
#             interv_logits = model.output.logits[:,-1].save() 
#             interv_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()

#         batch_intervention_ranks, _ = compute_token_rank_prob(interv_logits, batch_answer_tokens)  
#         batch_corrupt_ranks, _ = compute_token_rank_prob(corrupt_logits, batch_answer_tokens)  

#         # Calculate normalized logit diff
#         intervention_diff = interv_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         batch_effect = intervention_diff.mean() / baseline.mean()
#         logit_recovery += batch_effect * (current_batch_size / total_samples) # Accumulate batch results (weighted by batch size)

#         del interv_logits, corrupt_logits, clean_act_interface, act_execution # free memory
#         free_unused_cuda_memory()       
      
#         intervention_ranks.append(batch_intervention_ranks.to('cpu'))
#         corrupt_ranks.append(batch_corrupt_ranks.to('cpu'))

#     intervention_ranks = torch.cat(intervention_ranks)
#     corrupt_ranks = torch.cat(corrupt_ranks)
   
#     return intervention_ranks, corrupt_ranks, logit_recovery

# def eval_interface_necessity_rank_logit(
#     model: LanguageModel,
#     clean_prompts: list[str] = None,
#     corrupt_prompts: list[str] = None,
#     answers: list[str] = None,
#     interface_grouped_heads: dict[int, list[int]] = None, 
#     execution_grouped_heads: dict[int, list[int]] = None, 
#     batch_size: int = 8,
#     remote: bool = False,
#     prepend_bos:int = True,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
#     """ 
#     corrupt run: clean prompts 
#         cache corrupted activations of interface heads (corrupt_act_interface) 
#     patching run 1: clean prompts
#         patch in the corrupted activations of interface heads, cache activations of execution heads (corrupt_act_execution)
#     patching run 2: clean prompts 
#         patch in the cached corrupted activations of execution heads  

#     Args:
#         model: Language model to analyze
#         clean_prompts: List of prompts
#         corrupt_prompts: List of prompts
#         answers: List of expected answer strings
#         interface_grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. 
#         execution_grouped_heads: a dict where all heads of a particular layer are grouped under its layer key. 
#         batch_size: Size of each processing batch
#         remote: Whether to run model remotely

#     Returns:
#         intervention_ranks (torch.Tensor): ranks of the intervention tokens
#         corrupt_ranks (torch.Tensor): ranks of the corrupted tokens
#         clean_ranks (torch.Tensor): ranks of the clean tokens
#         rank_promotion (float): normalized rank difference between intervention and corruption
#     """
#     # Input validation
#     total_samples = len(clean_prompts)
#     if not (total_samples == len(corrupt_prompts) == len(answers)):
#         raise ValueError("All input lists must have the same length")

#     # Get model specs
#     spec = get_model_specs(model)
#     n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
#     d_head = d_model // n_heads

#     # Tokenize all data upfront
#     #print("Tokenizing all data...")
#     all_clean_tokens = model.tokenizer(
#         clean_prompts,
#         padding=True,
#         padding_side="left",
#         add_special_tokens=prepend_bos,
#         return_tensors="pt",
#     ).input_ids

#     all_corrupt_tokens = model.tokenizer(
#         corrupt_prompts,
#         padding=True,
#         padding_side="left",
#         add_special_tokens=prepend_bos,
#         return_tensors="pt",
#     ).input_ids

#     # Get token IDs for answers
#     answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

#     # Initialize results storage 
#     intervention_ranks = []
#     logit_diff = 0

#     # Process in batches
#     # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
#     for batch_start in range(0, total_samples, batch_size):
#         batch_end = min(batch_start + batch_size, total_samples)
#         current_batch_size = batch_end - batch_start

#         # Get current batch tokens from pre-tokenized data
#         clean_tokens = all_clean_tokens[batch_start:batch_end]
#         corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
#         batch_answer_tokens = answer_tokens[batch_start:batch_end]

#         # Get clean and corrupt logits at answer idx
#         clean_logits = model.trace(clean_tokens, trace=False, remote=remote)["logits"][:, -1]
#         corrupt_logits = model.trace(corrupt_tokens, trace=False, remote=remote)["logits"][:, -1]
#         clean_answer_logits = clean_logits[torch.arange(current_batch_size), batch_answer_tokens]
#         corrupt_answer_logits = corrupt_logits[torch.arange(current_batch_size), batch_answer_tokens]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)

#         # Step 1: Collect corrupted activations of interface heads
#         corrupt_act_interface = {}
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(corrupt_tokens):
#                 for layer in interface_grouped_heads.keys(): # only need to cache relevant layers' activation 
#                     attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in interface_grouped_heads[layer]: # only need to cache relevant heads' activation 
#                         corrupt_act_interface[(layer, head)] = attn_reshaped[:, head, :].save()

#         # Step 2: Clean runs, patch in the corrupted activations of interface heads and cache activations of execution heads 
#         corrupt_act_execution = {}
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(clean_tokens):
#                 for layer in interface_grouped_heads.keys():
#                     attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     attn_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in interface_grouped_heads[layer]: 
#                         # patch corrupted activations of interface heads
#                         attn_reshaped[:, head, :] = corrupt_act_interface[(layer, head)] 
#                 for layer in execution_grouped_heads.keys():
#                     exe_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     exe_reshaped = exe_attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in execution_grouped_heads[layer]: 
#                         # cache the activations of execution heads
#                         corrupt_act_execution[(layer, head)] = exe_reshaped[:, head, :].save()
                    
#         # Step 3: Clean runs, patch in the cached corrupted activations of execution heads 
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(clean_tokens):
#                 for layer in execution_grouped_heads.keys():
#                     interv_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
#                     interv_reshaped = interv_attn_output.reshape(current_batch_size, n_heads, d_head)
#                     for head in execution_grouped_heads[layer]: 
#                         # patch corrupted activation of execution heads
#                         interv_reshaped[:, head, :] = corrupt_act_execution[(layer, head)] 
#             interv_logits = model.output.logits[:,-1].save() 
#             interv_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()

#         batch_intervention_ranks, _ = compute_token_rank_prob(interv_logits, batch_answer_tokens)  

#         # Calculate normalized logit diff
#         intervention_diff = clean_answer_logits.to("cpu") - interv_answer_logits.to("cpu")
#         baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         batch_effect = intervention_diff.mean() / baseline.mean()
#         logit_diff += batch_effect * (current_batch_size / total_samples) # Accumulate batch results (weighted by batch size)

#         del interv_logits, corrupt_act_interface, corrupt_act_execution # free memory
#         free_unused_cuda_memory()       
      
#         intervention_ranks.append(batch_intervention_ranks.to('cpu'))
 
#     intervention_ranks = torch.cat(intervention_ranks)
   
#     return intervention_ranks, logit_diff

# def set_activation_neuron_rank_logit(
#     model: LanguageModel, batch_size: int = 8,
#     clean_prompts: list[str] = None, corrupt_prompts: list[str] = None, answers: list[str] = None,
#     all_clean_tokens: torch.Tensor = None, 
#     all_corrupt_tokens: torch.Tensor = None, 
#     all_answer_tokens: torch.Tensor = None,
#     grouped_neurons: dict[int, list[int]] = None, 
#     remote: bool = False, prepend_bos:int = True,
#     scaling_factor: float = 1.0, offset: float = None,
# ) -> torch.Tensor:
#     """ 

#     Args:
#         model: Language model to analyze
#         clean_prompts: List of prompts
#         corrupt_prompts: List of prompts
#         answers: List of expected answer strings
#         all_clean_tokens: Pre-tokenized clean tokens
#         all_corrupt_tokens: Pre-tokenized corrupt tokens
#         all_answer_tokens: Pre-tokenized answer tokens
#         grouped_neurons: a dict where all neurons, keyed by layer_index, value is a list of neuron_index. 
#         batch_size: Size of each processing batch
#         remote: Whether to run model remotely
       
#     Returns:
#         intervention_ranks (torch.Tensor): ranks of the intervention tokens
#     """
#     # Get model specs
#     spec = get_model_specs(model)
#     n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]
#     d_head = d_model // n_heads

#     # Tokenize all data upfront
#     if all_clean_tokens is None:
#         all_clean_tokens = model.tokenizer(
#             clean_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_corrupt_tokens is None:
#         all_corrupt_tokens = model.tokenizer(
#             corrupt_prompts,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=prepend_bos,
#             return_tensors="pt",
#         ).input_ids
#     if all_answer_tokens is None:
#         all_answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

#      # Input validation
#     total_samples = all_clean_tokens.shape[0]
#     if not (total_samples == all_corrupt_tokens.shape[0] == len(all_answer_tokens)):
#         raise ValueError("All input lists must have the same length")

#     # Initialize results storage 
#     intervention_ranks = []
#     logit_recovery = 0

#     # Process in batches
#     # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
#     for batch_start in range(0, total_samples, batch_size):
#         batch_end = min(batch_start + batch_size, total_samples)
#         current_batch_size = batch_end - batch_start

#         # Get current batch tokens from pre-tokenized data
#         batch_clean_tokens = all_clean_tokens[batch_start:batch_end]
#         batch_corrupt_tokens = all_corrupt_tokens[batch_start:batch_end]
#         batch_answer_tokens = all_answer_tokens[batch_start:batch_end]

#         # Get clean and corrupt logits at answer idx
#         clean_logits = model.trace(batch_clean_tokens, trace=False, remote=remote)["logits"][:, -1]
#         corrupt_logits = model.trace(batch_corrupt_tokens, trace=False, remote=remote)["logits"][:, -1]
#         clean_answer_logits = clean_logits[torch.arange(current_batch_size), batch_answer_tokens]
#         corrupt_answer_logits = corrupt_logits[torch.arange(current_batch_size), batch_answer_tokens]

#         # Get accessor for the model
#         accessor_config = get_accessor_config(model)
#         accessor = ModelAccessor(model, accessor_config)

#         # Collect clean activations of grouped neurons
#         clean_act = {}
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_clean_tokens):
#                 for layer in grouped_neurons.keys(): # only need to cache relevant layers' activation 
#                     neuron_act = accessor.layers[layer].mlp.down_proj.unwrap().input[:, -1]
#                     for neuron in grouped_neurons[layer]: # only need to cache relevant neurons' activation 
#                         clean_act[(layer, neuron)] = neuron_act[:, neuron].save()
#                 #clean_logits = model.output.logits[:,-1].save()
                    
#         # Corrupt runs, patch the clean activations of grouped neurons
#         with accessor.trace(remote=remote) as tracer:
#             with tracer.invoke(batch_corrupt_tokens):
#                 for layer in grouped_neurons.keys():
#                     interv_neuron_act = accessor.layers[layer].mlp.down_proj.unwrap().input[:, -1] # (batch_size, d_mlp)
#                     for neuron in grouped_neurons[layer]: 
#                         # patch clean activation of grouped neurons
#                         if offset is not None:
#                             interv_neuron_act[:, neuron] = clean_act[(layer, neuron)] + offset
#                         else:
#                             interv_neuron_act[:, neuron] = scaling_factor * clean_act[(layer, neuron)] 
                
#             # Get intervention logits
#             interv_answer_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(current_batch_size), batch_answer_tokens].save()
#             interv_logits = model.output.logits[:,-1].save() 

#         # Calculate normalized intervention effect on logits
#         intervention_diff = interv_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
#         batch_effect = intervention_diff.mean() / baseline.mean()
#         logit_recovery += batch_effect * (current_batch_size / total_samples) # Accumulate batch results (weighted by batch size)
        
#         batch_intervention_ranks, _ = compute_token_rank_prob(interv_logits, batch_answer_tokens) 

#         del interv_logits, corrupt_logits, clean_logits, clean_act # free memory
#         free_unused_cuda_memory()       
      
#         intervention_ranks.append(batch_intervention_ranks.to('cpu'))
    
#     intervention_ranks = torch.cat(intervention_ranks)
   
#     return intervention_ranks, logit_recovery

# def get_rank_logit_acc_raw_result_dict(
#     model: LanguageModel, 
#     top_heads: dict[int, dict[int, list[int]]]=None,
#     prompts: list[str]=None,
#     corrupt_prompts: list[str]=None,
#     answers: list[str]=None, batch_size: int = 8,
#     all_clean_tokens: torch.Tensor = None,
#     all_corrupt_tokens: torch.Tensor = None,
#     all_answer_tokens: torch.Tensor = None,
#     intervention_type: str = "set_component_to_logits",
# ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
#     """
#     Get the reciprocal rank, logit recovery, and accuracy for various number of top shared heads
#     Args:
#         intervention_type: type of intervention, 
#             "set_component_to_logits" or "set_component_to_logits_ablation" or
#             "set_activation" or "set_ablation" or 
#         top_heads: a dict of dicts, outer dict keyed by number of heads for intervention, 
#             inner dict keyed by layer (integer), value is a list of haead indices 
#     """
#     rank_dict = {}
#     logit_recovery_dict = {}
#     acc_dict = {}

#     for n_heads in top_heads.keys():
#         if intervention_type == "set_component_to_logits":
#             interv_ranks, logit_recovery = set_component_to_logits(
#             model=model, batch_size=batch_size,
#             clean_prompts=prompts,
#             corrupt_prompts=corrupt_prompts,
#             answers=answers,
#             all_clean_tokens=all_clean_tokens,
#             all_corrupt_tokens=all_corrupt_tokens,
#             all_answer_tokens=all_answer_tokens,
#             grouped_heads=top_heads[n_heads])
#         elif intervention_type == "set_activation":
#             interv_ranks, logit_recovery = set_activation_rank_logit(
#                 model=model, batch_size=batch_size,
#                 clean_prompts=prompts,
#                 corrupt_prompts=corrupt_prompts,
#                 answers=answers,
#                 grouped_heads=top_heads[n_heads])
#         elif intervention_type == "set_ablation":
#             interv_ranks, logit_recovery = set_ablation_rank_logit(
#                 model=model, batch_size=batch_size,
#                 clean_prompts=prompts,
#                 corrupt_prompts=corrupt_prompts,
#                 answers=answers,
#                 grouped_heads=top_heads[n_heads])
#         elif intervention_type == "set_component_to_logits_ablation":
#             raise NotImplementedError("set_component_to_logits_ablation not implemented")
#         else:
#             raise ValueError(f"Invalid intervention type: {intervention_type}")
        
#         interv_acc = compute_top_k_accuracy(interv_ranks.to('cpu'), 1)
#         reciprocal_rank = [1 / (x+1) for x in interv_ranks] # +1 to avoid division by zero
#         rank_dict[n_heads] = sum(reciprocal_rank) / len(reciprocal_rank)
#         logit_recovery_dict[n_heads] = logit_recovery
#         acc_dict[n_heads] = interv_acc

#     return rank_dict, logit_recovery_dict, acc_dict

# def get_rank_logit_acc_raw_result_dict_scaling(
#     model: LanguageModel,  batch_size: int = 20,
#     grouped_heads: dict[int, list]=None,
#     prompts: list[str]=None, corrupt_prompts: list[str]=None, answers: list[str]=None, 
#     all_clean_tokens: torch.Tensor = None,
#     all_corrupt_tokens: torch.Tensor = None,
#     all_answer_tokens: torch.Tensor = None,
#     scaling_factors: list[float]=None,
#     intervention_type: str = "set_component_to_logits",
#     average = False,
# ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
#     """
#     Get the reciprocal rank, logit recovery, and accuracy across different scaling factors
#     Args:
#         intervention_type: type of intervention, 
#         "set_component_to_logits" or "set_component_to_logits_ablation" or
#         "set_activation" or "set_ablation" or 
#         average: whether to use the average activations for patching 
#     """
#     rank_dict = {}
#     logit_recovery_dict = {}
#     acc_dict = {}

#     for scaling_factor in scaling_factors:
#         if intervention_type == "set_component_to_logits":
#             interv_ranks, logit_recovery = set_component_to_logits(
#             model=model, batch_size=batch_size,
#             clean_prompts=prompts, answers=answers,
#             corrupt_prompts=corrupt_prompts,
#             grouped_heads=grouped_heads, scaling_factor=scaling_factor,
#             all_clean_tokens=all_clean_tokens,
#             all_corrupt_tokens=all_corrupt_tokens,
#             all_answer_tokens=all_answer_tokens)
#         elif intervention_type == "set_activation":
#             if average:
#                 interv_ranks, logit_recovery = set_activation_rank_logit_average(
#                     model=model, batch_size=batch_size,
#                     all_clean_tokens=all_clean_tokens,
#                     all_corrupt_tokens=all_corrupt_tokens,
#                     all_answer_tokens=all_answer_tokens,
#                     grouped_heads=grouped_heads,
#                     scaling_factor=scaling_factor)
#             else:
#                 interv_ranks, logit_recovery = set_activation_rank_logit(
#                     model=model, batch_size=batch_size,
#                     clean_prompts=prompts, answers=answers, 
#                     corrupt_prompts=corrupt_prompts,
#                     all_clean_tokens=all_clean_tokens,
#                     all_corrupt_tokens=all_corrupt_tokens,
#                     all_answer_tokens=all_answer_tokens,
#                     grouped_heads=grouped_heads,
#                     scaling_factor=scaling_factor)
#         elif intervention_type == "set_ablation":
#             interv_ranks, logit_recovery = set_ablation_rank_logit(
#                 model=model, batch_size=batch_size,
#                 clean_prompts=prompts,corrupt_prompts=corrupt_prompts,
#                 answers=answers, grouped_heads=grouped_heads,
#                 all_clean_tokens=all_clean_tokens,
#                 all_corrupt_tokens=all_corrupt_tokens,
#                 all_answer_tokens=all_answer_tokens,
#                 scaling_factor=scaling_factor)
#         else:
#             raise ValueError(f"Invalid intervention type: {intervention_type}")
        
#         interv_acc = compute_top_k_accuracy(interv_ranks.to('cpu'), 1)
#         reciprocal_rank = [1 / (x+1) for x in interv_ranks] # +1 to avoid division by zero
#         rank_dict[scaling_factor] = sum(reciprocal_rank) / len(reciprocal_rank)
#         logit_recovery_dict[scaling_factor] = logit_recovery
#         acc_dict[scaling_factor] = interv_acc

#     return rank_dict, logit_recovery_dict, acc_dict

# def get_rank_logit_acc_raw_result_dict_scaling_neuron(
#     model=None, prompts=None, corrupt_prompts=None, 
#     answers=None, grouped_neurons=None, scaling_factors=None,
#     all_clean_tokens=None, all_corrupt_tokens=None, all_answer_tokens=None,
#     batch_size=20, 
# ):
#     rr_mean_dict = {}
#     acc_dict= {}
#     logit_recovery_mean_dict = {}
#     for scaling_factor in scaling_factors:
#         interv_ranks, logit_recovery = set_activation_neuron_rank_logit(
#             model=model, batch_size=batch_size,
#             clean_prompts=prompts, corrupt_prompts=corrupt_prompts, answers=answers,
#             all_clean_tokens=all_clean_tokens,
#             all_corrupt_tokens=all_corrupt_tokens,
#             all_answer_tokens=all_answer_tokens,
#             grouped_neurons=grouped_neurons,
#             scaling_factor=scaling_factor)
#         interv_acc = compute_top_k_accuracy(interv_ranks.to('cpu'), 1)
#         reciprocal_rank = [1 / (x+1) for x in interv_ranks] # +1 to avoid division by zero
#         rank_dict = sum(reciprocal_rank) / len(reciprocal_rank)
#         rr_mean_dict[scaling_factor] = rank_dict
#         acc_dict[scaling_factor] = interv_acc
#         logit_recovery_mean_dict[scaling_factor] = logit_recovery
#     return rr_mean_dict, logit_recovery_mean_dict, acc_dict

# def get_rank_logit_dict_neuron(
#     model=None,  batch_size=None,
#     prompts:list[str]=None, answers:list[str]=None,corrupt_prompts:list[str]=None,
#     all_clean_tokens:torch.Tensor=None, 
#     all_corrupt_tokens:torch.Tensor=None, 
#     all_answer_tokens:torch.Tensor=None,
#     top_neurons_dict:dict[int, dict]=None,
#     n_neurons_list:list[int]=None, index_list:list[int]=None,
   
# ):
#     rank_dict = {}
#     logit_recovery_dict = {}
#     acc_dict = {}
#     for n_top_neurons in n_neurons_list:
#         interv_ranks, logit_recovery = set_component_to_logits_neuron(
#             model=model, batch_size=batch_size,
#             clean_prompts=prompts, ccorrupt_prompts=corrupt_prompts, answers=answers,
#             all_clean_tokens=all_clean_tokens,
#             all_corrupt_tokens=all_corrupt_tokens,
#             all_answer_tokens=all_answer_tokens,
#             top_neurons=top_neurons_dict, n_top_neurons=n_top_neurons,
#             index_list=index_list,
#         )
    
#         interv_acc = compute_top_k_accuracy(interv_ranks.to('cpu'), 1)
#         reciprocal_rank = [1 / (x+1) for x in interv_ranks] # +1 to avoid division by zero
#         rank_dict[n_top_neurons] = sum(reciprocal_rank) / len(reciprocal_rank)
#         logit_recovery_dict[n_top_neurons] = logit_recovery
#         acc_dict[n_top_neurons] = interv_acc
#     return rank_dict, logit_recovery_dict, acc_dict

# def get_rank_logit_acc_result_dict_necessity(
#     model: LanguageModel,
#     top_shared_heads: dict[int, dict[int, list[int]]],
#     prompts: list[str],
#     corrupt_prompts: list[str],
#     answers: list[str],
# ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
#     """
#     Get the rank demotion, logit degradation, and accuracy recovery for various number of top shared heads
#     """
#     rank_demotion_dict = {}
#     logit_degradation_dict = {}
#     acc_degradation_dict = {}

#     for n_shared_heads in top_shared_heads.keys():
#         interv_ranks, clean_ranks, rank_demotion, logit_degradation = set_ablation_rank_logit(
#             model=model,
#             clean_prompts=prompts,
#             corrupt_prompts=corrupt_prompts,
#             answers=answers,
#             grouped_heads=top_shared_heads[n_shared_heads])
#         interv_acc = compute_top_k_accuracy(interv_ranks.to('cpu'), 1)
#         clean_acc = compute_top_k_accuracy(clean_ranks.to('cpu'), 1)
#         acc_degradation =  clean_acc - interv_acc 
#         rank_demotion_dict[n_shared_heads] = rank_demotion
#         logit_degradation_dict[n_shared_heads] = logit_degradation
#         acc_degradation_dict[n_shared_heads] = acc_degradation

#     return rank_demotion_dict, logit_degradation_dict, acc_degradation_dict


