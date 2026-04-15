import torch 
from nnsight import LanguageModel
from Shared_utils.wrapper import get_model_specs
from Shared_utils.shared_utils import sample_random_heads

def get_vector_from_universal_top_indirect_effect_heads(model:LanguageModel, 
    model_name: str = None,
    n_top_heads: int = 10,
    mean_activations: torch.Tensor = None,
    device: str = "cuda:0",
    random_heads: bool = False,
    seed: int = None,
    head_source: str = "EP",
):
    """
    Get the vector from the sum of theoutput of heads that have biggest average indirect effect across datasets.
    Args: 
         random_heads: list of the random heads represented as tuples [(L,H,S), ...], 
            (L=Layer, H=Head, S=random Effect Score) #TODO: validate this 
        head_source: head extracted from "EP" or "IP" runs 
    """
    if random_heads:
        top_heads = sample_random_heads(model, num_samples=n_top_heads, seed=seed)
    else: 
        if model_name == 'Llama-3.1-8B-Instruct' :
            if head_source == "EP":
                top_heads = [(13, 27, 0.1256), (16, 29, 0.0297), (15, 17, 0.0292), (10, 5, 0.0221), (15, 1, 0.0205), 
                            (15, 16, 0.02), (10, 7, 0.0139), (30, 16, 0.0105), (15, 28, 0.0103), (11, 30, 0.0095), 
                            (11, 29, 0.0091), (14, 11, 0.0079), (11, 28, 0.0074), (15, 2, 0.0071), (14, 6, 0.0069), 
                            (17, 5, 0.0069), (10, 12, 0.0069), (31, 31, 0.0065), (13, 23, 0.006), (31, 14, 0.0056),
                ]
            elif head_source == "IP":
                raise NotImplementedError("IP heads not implemented for Llama-3.1-8B-Instruct")
        elif model_name == 'Llama-3.2-1B-Instruct':
            if head_source == "EP":
                top_heads = [(7, 28, 0.1317), (9, 8, 0.095), (7, 5, 0.0851), (7, 30, 0.0687), (9, 16, 0.047), 
                            (7, 6, 0.0463), (10, 16, 0.0328),(7, 9, 0.0326), (9, 7, 0.015), (8, 31, 0.0148), 
                            (8, 11, 0.0105), (8, 6, 0.01),(10, 15, 0.0086), (9, 13, 0.0082), (10, 23, 0.0079),
                ]
            elif head_source == "IP":
                raise NotImplementedError("IP heads not implemented for Llama-3.2-1B-Instruct")
        else:
            raise ValueError("Model not supported")

    spec = get_model_specs(model)
    n_heads, d_model = spec["n_heads"], spec["d_model"]
    d_head = d_model // n_heads

    function_vector = torch.zeros((1, d_model)).to(device)

    for L,H,_ in top_heads[:n_top_heads]:
        if 'gpt2-xl' in model_name.lower():
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_name.lower():
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'llama' in model_name.lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'gpt-neox' in model_name.lower():
            out_proj = model.gpt_neox.layers[L].attention.dense
        elif 'olmo' in model_name.lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        else: 
            raise ValueError("Model not supported")
        
        x = torch.zeros(d_model)
        x[H*d_head:(H+1)*d_head] = mean_activations[L,H]
        d_out = out_proj(x.reshape(1,d_model).to(device).to(model.dtype))

        function_vector += d_out
        function_vector = function_vector.to(model.dtype)
    
    return function_vector, top_heads