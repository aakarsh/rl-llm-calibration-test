import datasets as hugging_face_datasets
import numpy as np

import llm_calibration.runner.multiple_choice_questions as mcq
from llm_calibration.runner.multiple_choice_questions import  (run_inference) 
                                
def load_dataset(name="lucamccabe/logiqa", split='train'):
    return hugging_face_datasets.load_dataset(name)[split]

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
                  n_shots=1):
    return mcq.run_inference(model, tokenizer, dataset,
                         tag=tag, 
                         include_prompt=include_prompt, 
                         dataset_item_parser=parse_dataset_item,
                         alphanumeric_options=alphanumeric_options,
                         verbose=verbose, 
                         n_shots=n_shots)