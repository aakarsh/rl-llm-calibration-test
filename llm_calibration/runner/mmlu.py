import datasets as hugging_face_datasets
import numpy as np

import  llm_calibration.runner.multiple_choice_questions as mcq
from llm_calibration.model.model_probability import  get_log_prob_of_completion

DATASET_GROUPS = {
  "STEM": [
      'abstract_algebra',
      'anatomy',
      'astronomy',
      'college_biology',
      'college_chemistry',
      'college_computer_science',
      'college_mathematics',
      'college_physics',
      'computer_security',
      'conceptual_physics',
      'electrical_engineering',
      'elementary_mathematics',
      'high_school_biology',
      'high_school_chemistry',
      'high_school_computer_science',
      'high_school_mathematics',
      'high_school_physics',
      'high_school_statistics',
      'machine_learning'
  ],
  "HUMANITIES": [
      'formal_logic',
      'high_school_european_history',
      'high_school_us_history',
      'high_school_world_history',
      'human_sexuality',
      'international_law',
      'jurisprudence',
      'logical_fallacies',
      'moral_disputes',
      'moral_scenarios',
      'philosophy',
      'prehistory',
      'professional_law',
      'world_religions'
  ],
  "SOCIAL_SCIENCE": [
    'econometrics',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_microeconomics',
    'high_school_psychology',
    'human_sexuality',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy'
  ],
  "OTHER": [
    'business_ethics',
    'clinical_knowledge',
    'college_medicine',
    'global_facts',
    'human_aging',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'nutrition',
    'professional_accounting',
    'professional_medicine',
    'virology',
  ]
}

# TODO Use proto class
def load_dataset(name=None, split="test"):
  """
  Load the dataset, by either its name or its group name. 
  """
  dataset_path = 'cais/mmlu'
  datasets = []
  if name in DATASET_GROUPS: # allow groups of datasets concatenated together.
        datasets = [ hugging_face_datasets.load_dataset(dataset_path, dataset)[split] 
                      for dataset in DATASET_GROUPS[name] ]
  elif name is not None: # built in dataset
    datasets = [ hugging_face_datasets.load_dataset(dataset_path, name)[split] ]
  else: # load all datasets
    datasets = [ hugging_face_datasets.load_dataset(dataset_path,'all')[split] ]
  return hugging_face_datasets.concatenate_datasets(datasets, split=split)

def parse_dataset_item(item):
    return {
        "question": item["question"],
        "choices": item["choices"],
        "answer": item["answer"] 
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
                         start_idx=0
                         write_chunks=True,
                         chunk_size=100,
                         output_dir=output_dir,
                         verbose=verbose, 
                         n_shots=n_shots) 
