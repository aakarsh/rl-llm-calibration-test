"""
!pip install transformers torch datasets accelerate
# Needed for quantization
!pip install bitsandbytes-cuda110 bitsandbytes
# Probably not neccessary
!pip install numpy pandas
#"""

# import libraries
import sys
import numpy as np
import pandas as pd
import json
import random
import pathlib


import torch
import transformers
import datasets
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

#import bitsandbytes as bnb

from huggingface_hub import login
login(token="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")

def load_model(model_name="meta-llama/Llama-2-7b-chat-hf", quantized=True):
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
    # Also try : meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-7b-hf
    #active_model= "meta-llama/Llama-2-7b-chat-hf"
    #active_model= "meta-llama/Llama-2-7b-hf"
    active_model = model_name
    
    config = transformers.AutoConfig.from_pretrained(active_model)
    # Explicitly set the max_seq_len
    config.max_seq_len = 512
    config.max_answer_len= 10
    tokenizer = transformers.AutoTokenizer.from_pretrained(active_model)
    model = None
    if not quantized:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            active_model,
            config=config,
            # This requires accelerate to be installed
            #device_map="auto",
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            active_model,
            config=config,
            # This requires accelerate to be installed
            device_map="auto",
            # This require bits and bytes
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    return {"model_name": active_model, "model": model, "tokenizer": tokenizer}

def get_log_prob_of_completion(
        model,
        tokenizer,
        prompt,
        completion,
        # TODO: mps device? (apple silicon)
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    """
    Convenience function for computing the log probability of a completion
    given a prompt.
    """
    # tokenize the prompt and the completion
    # truncate so as to fit into to maximal context window of gpt-2
    # which is 1024 tokens
    input_ids = tokenizer(
        prompt + completion,
        return_tensors='pt',
        truncation=True,
        max_length=1024,
    )['input_ids'].to(device)
    # create attention mask and position ids
    attention_mask = (input_ids != tokenizer.eos_token_id).to(dtype=torch.int64)
    position_ids = attention_mask.cumsum(-1)-1
    # get the logits for the completion
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
    # get the logits of the completion
    # for that, make a tensor of the logits
    # for the completion only
    # in particular, we shift the indices by one to
    # the left to access logits of the
    # actual sequence tokens
    logits_completion = out.logits[:, :-1]
    logits_completion = logits_completion.squeeze()
    # get the log probabilities for the completion
    log_probs = torch.nn.functional.log_softmax(
        logits_completion,
        dim=-1
    )
    # retrieve the logit corresponding to the actual completion tokens
    try:
        log_completion_tokens = log_probs.gather(
            dim=-1,
            index=input_ids[:, 1:].squeeze().unsqueeze(-1)
        )
    except:
        log_completion_tokens = log_probs.gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        )
    # separately tokenize prompt
    # so as to access the logits for the completion only
    # when scoring the completion
    input_ids_prompt = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=1024
    )['input_ids'].to(device)
    prompt_end_index = input_ids_prompt.shape[-1] - 1
    continuationConditionalLogProbs = log_completion_tokens[
        prompt_end_index:
    ]
    completion_log_prob = torch.sum(
        continuationConditionalLogProbs
    ).cpu()
    return completion_log_prob

# TODO: Few shot
def trivia_qa_questions(dataset):
  split = "train"
  wrong_answers = [rec["answer"]["value"] for rec in dataset[split]]
  questions = []
  nalternatives = 3
  for rec in dataset[split]:
      #if ("answer" not in rec) or ("normalized_value" not in rec["answer"]):
      #  continue
    choices = ([rec["answer"]["value"]] +
               random.sample(wrong_answers, nalternatives))
    correct_choice = [True]+[False]*nalternatives
    shuffled = list(zip(choices, correct_choice))
    random.shuffle(shuffled)
    choices, correct_choice = zip(*shuffled)
    choice_names = choices
    choices = []
    prompt = rec["question"] + "\n"
    for letter, choice in zip("ABCDEFGHIJKLMOPQRS", choice_names):
        prompt += "  ({}) {}\n".format(letter, choice)
        choices.append("("+letter+")")
    prompt += "Answer: "
    questions.append({"prompt":prompt,
                      "choices": choices,
                      "choice_names": choice_names,
                      "correct_choice": correct_choice})
  return questions

def dump_questions(questions, file_name="trivia_qa-questions.json"):
    with open(file_name, 'w', encoding="utf-8") as fout:
        json.dump(questions, fout, indent="\t")

def load_questions(file_name="trivia_qa-questions.json"):
    questions = None
    with open(file_name, 'r', encoding="utf-8") as fin:
        questions = json.load(fin)
    return questions

def question_probs(model, question):
    res = []
    prompt = question["prompt"]
    nchoices = len(question["choices"])
    for correct, choice in zip(question["correct_choice"], question["choices"]):
        raw_prob = get_prob_of_completion(
            model = model["model"],
            tokenizer = model["tokenizer"],
            prompt = prompt,
            completion = choice
        )
        res.append({"prompt": prompt,
                    "choice": choice,
                    # Extract value from single element tensor
                    "raw_prob": raw_prob.item(),
                    "label": 1 if correct else 0})
    # Normalize
    raw_probs = [choice["raw_prob"] for choice in res]
    total = sum(raw_probs)
    correct = np.argmax(raw_probs)
    for i, choice in enumerate(res):
        choice["norm_prob"] = choice["raw_prob"] / total
        choice["prediction"] = 1 if i == correct else 0
    return res

def run_on_questions(model, questions):
    res = []
    for i, q in enumerate(questions):
        res += question_probs(model, q)
    return res


def get_prob_of_completion(model, tokenizer, prompt, completion):
    return torch.exp(get_log_prob_of_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt, completion=completion))


def run(dump_start=0, dump_step=250, dump_end=2000,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        file_prefix="trivia_qa-llama-2-7b-chat",
        quantized=True,
        question_dump="trivia_qa-questions.json"):
    print("=== Loading Model")
    model = load_model(model_name=model_name, quantized=quantized)

    print("=== Loading data")
    qdump = pathlib.Path(question_dump)
    questions = None
    if qdump.is_file():
        questions = load_questions(file_name=question_dump)
    else:
        data = datasets.load_dataset("mandarjoshi/trivia_qa", name="rc.nocontext")
        print("    --- transforming data")
        questions = trivia_qa_questions(data)
        dump_questions(questions)
        data = None
    print("   --- sanity check: ")
    print(questions[:5])
    print("=== running inference")
    i_prev = dump_start
    for i in range(i_prev+dump_step, dump_end+1 if dump_end > 0 else len(questions), dump_step):
        results = run_on_questions(model, questions[i_prev:i])
        output = [{"raw_prob": choice["raw_prob"],
                   "prob": choice["norm_prob"],
                   "label": choice["label"],
                   "prediction": choice["prediction"]}
                  for choice in results]
        fname = "{}-{:06}-{:06}.json".format(file_prefix, i_prev, i)
        with open(fname, 'w', encoding="utf-8") as fout:
            json.dump(output, fout, indent="\t")
        print("   --- wrote predictions {}-{}".format(i_prev, i))
        i_prev = i
    print("=== done.")


if __name__ == "__main__":
    run()
    
