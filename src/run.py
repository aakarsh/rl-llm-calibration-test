import torch
import json
import np
from torch.utils.data import DataLoader

import argparse
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
import torch

from huggingface_hub import login
from model.model_loader import load_model
from model.model_probability import (get_normalized_probabilities)
import dataset.mmlu as mmlu
from plot import plot_calibration

print("accelerate", is_accelerate_available())
print("is_bitsandbytes_available", is_bitsandbytes_available())

login(token="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")

def main():
    parser = argparse.ArgumentParser(description="Run PyTorch model on a dataset")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Path to the saved PyTorch model file")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True, default="high_school_world_history",
                        help="Dataset name")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="result.json",
                        help="Path to save the predictions JSON file (default: result.json)")

    args = parser.parse_args()
    
    # Load your model
    model, tokenizer = load_model(name=args.model_name)
    dataset = mmlu.load_dataset(name=args.dataset)
    model_results, mmlu_prediction_probabilities, mmlu_target_labels = \
        mmlu.run_inference(model, tokenizer, dataset)

    model_results = model_results[0]['results']
    completion_probabilities, truth_values = get_normalized_probabilities(model_results)

  
    plot_calibration(np.array(completion_probabilities), 
                  np.array(truth_values, dtype=np.int32), 
                  num_bins=20, range_start=0, range_end=1,
                  out_file="../output/calibration_"+args.model_name".png")
    # Create a dictionary to store results (modify as needed)

    # Save the results to a JSON file
    with open(args.output_file, "w") as f:
        json.dump("../output/model_results_"+args.model_name+".json", f, indent=4)

    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
