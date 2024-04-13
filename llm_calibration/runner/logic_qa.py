import datasets as hugging_face_datasets
import numpy as np

import llm_calibration.runner.multiple_choice_questions as mcq
from llm_calibration.runner.multiple_choice_questions import  (run_inference) 
                                
def load_dataset(name=None, split='train'):
    return hugging_face_datasets.load_dataset("lucasmccabe/logiqa")[split]

def parse_dataset_item(item):
    context = item["context"]
    query = item["query"]
    options = item["options"]
    
    question = f"{context}\nQuestion:{query}".format(context=context, query=query)
    return {
        "question": question,
        "choices": options,
        "answer": item["correct_option"]
    }
    
def run_inference(model, tokenizer, dataset,
                  tag="default_tag", include_prompt=False, 
                  alphanumeric_options = ['A', 'B', 'C', 'D'],
                  verbose = False, 
                  start_idx=0,
                  stop_idx=-1,
                  chunk_size=100,
                  write_chunks=False,
                  output_dir="",
                  n_shots=1):
    return mcq.run_inference(model, tokenizer, dataset,
                         tag=tag, 
                         include_prompt=include_prompt, 
                         dataset_item_parser=parse_dataset_item,
                         alphanumeric_options=alphanumeric_options,
                         verbose=verbose, 
                         start_idx=start_idx,
                         stop_idx=stop_idx,
                         chunk_size=chunk_size,
                         write_chunks=write_chunks,
                         output_dir=output_dir,
                         n_shots=n_shots)