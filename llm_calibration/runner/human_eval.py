import datasets as hugging_face_datasets
import numpy as np

import llm_calibration.runner.true_false_questions as tfq
                                
def load_dataset(name='train'):
    return hugging_face_datasets.load_dataset("openai_humaneval")

   
def run_inference(model, tokenizer, dataset,
                  tag="default_tag", include_prompt=False, 
                  alphanumeric_options = ['A', 'B'],
                  verbose = False, 
                  n_shots=1):
    return tfq.run_inference(model, tokenizer, dataset,
                         tag=tag, 
                         include_prompt=include_prompt, 
                         # dataset_item_parser=parse_dataset_item,
                         alphanumeric_options=alphanumeric_options,
                         verbose=verbose, 
                         n_shots=n_shots)