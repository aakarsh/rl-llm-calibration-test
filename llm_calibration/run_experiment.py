import logging
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
import llm_calibration.runner.logic_qa as logic_qa_runner 
import llm_calibration.runner.truthful_qa as truthful_qa_runner 
import llm_calibration.runner.human_eval as human_eval_runner 

from llm_calibration.plot import plot_calibration

from huggingface_hub import login

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

logger = logging.getLogger(__name__)


login(token="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")

LLAMA_MODELS = [
    'meta-llama/Llama-2-13b-hf', 
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-7b-hf'
    'meta-llama/Llama-2-70b-hf', 
    'meta-llama/Llama-2-70b-chat-hf',
] 

RUNNERS = {
    'mmlu': mmlu_runner,
    'logic_qa': logic_qa_runner,
    'human_eval': human_eval_runner,
    'truthful_qa' :truthful_qa_runner
}

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
    return "model_"+sanitize(model_name)+"_ds_"+sanitize(dataset_name)+"_n_shots_"+(str(n_shots))+"_tag"

def run_experiment(model_name, dataset_name, runner, output_dir, 
                   n_shots=1, 
                   write_chunks=True,
                   start_idx=0):
    """
    """
    logger.info(f"Running experiment with model {model_name} on dataset {dataset_name}")
    tokenizer, model = load_model(name=model_name)
    dataset = runner.load_dataset(name=dataset_name)

    model_tag = generate_model_tag(model_name, dataset_name, n_shots=n_shots)
    model_results, _, _ = \
        runner.run_inference(model, tokenizer, dataset, 
                           tag=model_tag,
                           include_prompt=False, 
                           write_chunks=write_chunks,
                           chunk_size=100,
                           start_idx=start_idx,
                           output_dir=output_dir,
                           n_shots=n_shots)

    # TODO Move output generation into runner.
    # Save the results to a JSON file, 
    # This need to be done as part of the inference.
    output_file_name = "model_results_"+model_tag+"-result.json"
    output_file = output_dir+"/"+output_file_name
    with open(output_file, "w") as f:
        json.dump(model_results, f, indent=4)

    logger.info(f"Model results saved to {output_file}")
    logger.info(f" Model Result summary:")
    pretty_print_model_results(model_results)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run PyTorch model on a dataset")

    # Model arguments
    parser.add_argument("--model_name", type=str,  default="meta-llama/Llama-2-7b-chat-hf",
                        help="Path to the saved PyTorch model file")
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default='', help="Dataset name")

    parser.add_argument("--runner_name", type=str,  help="name of inference runner")
     
    parser.add_argument("--n-shots", type=int,default=1 , help=1)

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Path to save the predictions JSON file (default: result.json)")

    parser.add_argument("--write-chunks", type=bool, default=True,
                        help="Path to save the predictions JSON file (default: result.json)")

    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index to resume execution from")


    args = parser.parse_args()
    run_experiment(args.model_name, 
                   args.dataset, RUNNERS[args.runner_name], 
                   args.output_dir, 
                   start_idx=args.start_idx, 
                   write_chunks=args.write_chunks,
                   n_shots=args.n_shots
                   )    

if __name__ == "__main__":
    logging.basicConfig(filename='experiment.log', 
                        level=logging.INFO, 
                        format='%(asctime)s %(message)s')
    main()
