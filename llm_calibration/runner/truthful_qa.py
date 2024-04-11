# TODO: Add Truthful QA: https://huggingface.co/datasets/truthful_qa?row=0A
import datasets as hugging_face_datasets
import numpy as np
import random
import  llm_calibration.runner.multiple_choice_questions as mcq


def load_dataset(name=None, split="test"):
  """
  Load the dataset, by either its name or its group name. 
  """
  dataset_path = 'truthful_qa' 
  datasets = [ hugging_face_datasets.load_dataset(dataset_path)[split] ]
  return hugging_face_datasets.concatenate_datasets(datasets, split=split)

def parse_dataset_item(item):
    choices = item["correct_answers"]+item["incorrect_answers"]
    # random.shuffle(mylist)

    return {
        "question": item["question"],
        "choices": item["correct_answers"]+item["incorrect_answers"],
        "answer": item["best_answer"] 
    }
    
def run_inference(model, tokenizer, dataset,
                  tag="default_tag", include_prompt=False, 
                  alphanumeric_options = ['A', 'B', 'C', 'D'],
                  verbose = False, 
                  start_idx=0,
                  write_chunks=True,
                  chunk_size=100,
                  output_dir=None,
                  n_shots=1):
    return mcq.run_inference(model, tokenizer, dataset,
                         tag=tag, 
                         include_prompt=include_prompt, 
                         dataset_item_parser=parse_dataset_item,
                         alphanumeric_options=alphanumeric_options,
                         start_idx=start_idx,
                         write_chunks=write_chunks,
                         chunk_size=100,
                         output_dir=output_dir,
                         verbose=verbose, 
                         n_shots=n_shots) 
