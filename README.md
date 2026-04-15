# Lexical_Task_Representations

## Behavior resutls 
Example command for running the behavior variance analysis:
python behavior_variance.py \
        --model_name="Qwen/Qwen3-4B-Thinking-2507"  \
        --d_name="country-capital" \
        --prompt_type="EP"

## Identify lexical task heads 
Example command for identifying lexical task heads:
python identify_heads_nnsight.py \
        --model_name="Qwen/Qwen3-4B-Thinking-2507" \
        --d_name="country-capital" \
        --prompt_type="EP" \
        --component_type="lexical_task"

## Causal intervention 
Example command for causal interventions to fix incorrect prompts:
python causal_fix_incorrect.py \
        --model_name="Qwen/Qwen3-4B-Thinking-2507" \
        --d_name="country-capital" \
        --prompt_type_fix="EP" \
