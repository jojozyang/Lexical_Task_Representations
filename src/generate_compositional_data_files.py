import json
import os
from datasets import load_dataset
from collections import defaultdict

def generate_task_json_files(dataset_name="apoorvkh/composing-functions"):
    # 1. Load the dataset (defaulting to the 'train' split)
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split='train')

    # 2. Group data by task
    # We use a dictionary where keys are task names and values are lists of formatted dictionaries
    tasks_data = defaultdict(list)

    print("Processing rows...")
    for row in dataset:
        task_name = row['task']
        
        # Structure for GFx
        entry_gfx = {
            "input": row['x'],
            "output": row['GFx']
        }
        
        # Add entries to the specific task group
        tasks_data[task_name].append(entry_gfx)

    # 3. Create a directory for the output if it doesn't exist
    output_dir = "../datasets/compositional"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4. Save each task to its own .json file
    print(f"Saving files to '{output_dir}/'...")
    for task_name, entries in tasks_data.items():
        # Sanitize task name for filename (remove spaces/special characters if necessary)
        safe_filename = "".join([c if c.isalnum() else "_" for c in task_name])
        file_path = os.path.join(output_dir, f"{safe_filename}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=4)
    
    print("Done! All task files have been created.")

if __name__ == "__main__":
    generate_task_json_files()