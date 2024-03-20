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

def load_model(name="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name,
                load_in_4bit=True,
                device_map="auto",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
    return tokenizer, model