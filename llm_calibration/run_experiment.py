import torch
import json
import numpy as np
from torch.utils.data import DataLoader

import argparse

from huggingface_hub import login
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import evaluate
import torch

from llm_calibration.model.model_probability import (get_normalized_probabilities, pretty_print_model_results)
import llm_calibration.runner.mmlu as mmlu_runner
from llm_calibration.plot import plot_calibration

from huggingface_hub import login

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

login(token="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")

LLAMA_MODELS = [
    'meta-llama/Llama-2-13b-hf', 
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-7b-hf'
    'meta-llama/Llama-2-70b-hf', 
    'meta-llama/Llama-2-70b-chat-hf',
] 

SUPPORTED_MODELS = [] + LLAMA_MODELS

def load_model(name="meta-llama/Llama-2-7b-chat-hf"):
    """
    Add all models which we support loading and running here with default parameters.
    
    Returns:
        tokenizer: Tokenizer for the model
        model: Model to run inference on
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {name} not supported. Supported models are {SUPPORTED_MODELS}") 
    if name in LLAMA_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name,
                    load_in_4bit=True,
                    device_map="auto",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16)
        return tokenizer, model
    else:
        raise ValueError(f"Model {name} not supported. Supported models are {SUPPORTED_MODELS}")

def generate_model_tag(model_name, dataset_name, n_shots=1, include_date=False):
    """
    Generate a tag which uniquely describes a model dataset experiment.
    """
    def sanitize(s): 
        'remove special characters from a string'
        return s.replace("/", "_")
    return "model_"+sanitize(model_name)+"_ds_"+sanitize(dataset_name)+"_n_shots_"+(str(n_shots))+"_tag",

def run_experiment(model_name, dataset_name, runner, output_dir, output_tag, n_shots=1):
    """
        - Thinking-Step-By-Step.
    """
    tokenizer, model = load_model(name=model_name)
    dataset = runner.load_dataset(name=dataset_name)

    model_tag = generate_model_tag(model_name, dataset_name, n_shots=n_shots)
    model_results, _, _ = \
        runner.run_inference(model, tokenizer, dataset, 
                           tag=model_tag,
                           include_prompt=False, n_shots=n_shots)

    # Save the results to a JSON file
    output_file = output_dir+"model_results_"+output_tag+"-result.json"
    with open(output_file, "w") as f:
        json.dump(model_results, f, indent=4)

    pretty_print_model_results(model_results)
    return True


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
