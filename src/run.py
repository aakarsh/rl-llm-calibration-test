from transformers.utils import is_accelerate_available, is_bitsandbytes_available
# import libraries
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import torch
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import evaluate

from huggingface_hub import login

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch
from datasets import (
    load_dataset,
    Dataset
)

print("accelerate", is_accelerate_available())
print("is_bitsandbytes_available", is_bitsandbytes_available())

login(token="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")

active_model= "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(active_model)
model = AutoModelForCausalLM.from_pretrained(active_model,
  load_in_4bit=True,
  device_map="auto",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.float16)


if __name__ == "__main__":
  # Given a model and dataset name we run start an inference job producing a
  # model response output.
  
  print("model", model)
  print("tokenizer", tokenizer)
  print("model.device", model.device)
  print("model.device.type", model.device.type)
  
