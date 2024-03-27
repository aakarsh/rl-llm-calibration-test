import torch
import json
import numpy as np
from torch.utils.data import DataLoader

import argparse
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
import torch

from huggingface_hub import login

from llm_calibration.model.model_loader import load_model
from llm_calibration.model.model_probability import (get_normalized_probabilities)
import llm_calibration.runner.mmlu as mmlu_runner
from llm_calibration.plot import plot_calibration

print("accelerate", is_accelerate_available())
print("is_bitsandbytes_available", is_bitsandbytes_available())

login(token="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")


def run_experiment(model_name, dataset_name, runner, output_dir, output_tag, n_shots=1):
    """
    TODO:
        - 5-shot prompting 
        - Subject-wise calibration plots
        - Thinking-Step-By-Step.
    """
    sanitized_model_name = model_name.replace("/", "_") 
    sanitized_dataset_name = dataset_name.replace("/", "_")
    
    tokenizer, model = load_model(name=model_name)
    dataset = runner.load_dataset(name=dataset_name)
    
    model_results, _, _ = \
        runner.run_inference(model, tokenizer, dataset, 
                           tag="model_"+sanitized_model_name+"_ds_"+sanitized_dataset_name+"_n_shots_"+(str(n_shots))+"_tag", 
                           include_prompt=False, n_shots=n_shots)
        
    # Save the results to a JSON file
    output_file = output_dir+"model_results_"+output_tag+"-result.json"
    with open(output_file, "w") as f:
        json.dump(model_results, f, indent=4)

    completion_probabilities, truth_values = get_normalized_probabilities(model_results)
  
    plot_calibration(np.array(completion_probabilities), 
                    np.array(truth_values, dtype=np.int32), 
                    num_bins=10, range_start=0, range_end=1,
                    out_file=output_dir+"/calibration_"+output_tag+".png")
    
    return model_results, completion_probabilities, truth_values

def main():
    parser = argparse.ArgumentParser(description="Run PyTorch model on a dataset")

    # Model arguments
    parser.add_argument("--model_name", type=str,  default="meta-llama/Llama-2-7b-chat-hf",
                        help="Path to the saved PyTorch model file")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="high_school_world_history",
                        help="Dataset name")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Path to save the predictions JSON file (default: result.json)")

    parser.add_argument("--output-tag", type=str, default="llama",
                        help="Path to save the predictions JSON file (default: result.json)")

    args = parser.parse_args()

    run_experiment(args.model_name, args.dataset, mmlu_runner, args.output_dir, args.output_tag)    

if __name__ == "__main__":
    main()
