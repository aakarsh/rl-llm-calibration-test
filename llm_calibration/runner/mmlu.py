import datasets as hugging_face_datasets
import numpy as np

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
def load_dataset(name, split="test"):
  """
  Load the dataset, by either its name or its group name. 
  """
  dataset_path = 'cais/mmlu'
  datasets = []
  if name in DATASET_GROUPS: # allow groups of datasets concatenated together.
        datasets = [ hugging_face_datasets.load_dataset(dataset_path, dataset)[split] 
                      for dataset in DATASET_GROUPS[name] ]
  else: # built in dataset
    datasets = [ hugging_face_datasets.load_dataset(dataset_path, name)[split] ]
  return hugging_face_datasets.concatenate_datasets(datasets, split=split)

# TODO pull out to utility
def generate_n_shot_prompt(dataset,
                           question_idx,
                           prompt_template="{question}",
                           options_template="({choice})",
                           answer_template="Answer:{answer}",
                           n_shots=5):
  """
  Generate a n-shot prompt.
  """
  def format_question(current_idx, with_answer=True):
    item = dataset[current_idx]
    question_template = prompt_template.format(question=item["question"])
    choices_template = "\n"+"\n".join(["%s. %s" % (options_template.format(choice=alpha), choice) for alpha, choice in zip(alphanumeric_options, item['choices'])])
    question_prompt = "%s\n%s\n" % (question_template, choices_template)
    if with_answer:
      question_prompt += "\n"+ answer_template.format(answer=options_template.format(choice=alphanumeric_options[item['answer']]))+"\n"
    else:
      question_prompt += "\n"+ answer_template.format(answer="")
    return question_prompt
 
  alphanumeric_options = ['A', 'B', 'C', 'D']
  formatted_options = ["(%s)" % choice for choice in alphanumeric_options ]
  current_idx = 0 
  question_buffer = ""

  while n_shots > 1:
    if current_idx >= len(dataset): 
      break
    if not current_idx == question_idx:
      question_buffer += format_question(current_idx) +"\n"
    current_idx += 1
    n_shots -= 1
  question_buffer += format_question(question_idx, with_answer=False)
  return question_buffer, formatted_options

def run_single_inference(model, tokenizer, prompt, selections, item, verbose=False):
    """
    Run single inference.
    """
    if verbose: print(prompt)

    selection_log_prob_opt_option = []
    for _, selection  in enumerate(selections):
      # run inference here for each selection.
      log_prob_opt_option = get_log_prob_of_completion(model=model, tokenizer=tokenizer, 
                                                       prompt=prompt, completion=selection)
      if verbose:
        print("selection",selection, "log_prob:",log_prob_opt_option)

      selection_log_prob_opt_option.append(np.float64(log_prob_opt_option.detach().numpy()))
    
    selection_results = (dict(zip(selections, np.float64(selection_log_prob_opt_option))))
    chosen_selection = np.argmax(selection_log_prob_opt_option)
    # dataset specific.
    alphanumeric_options = ['A', 'B', 'C', 'D'] 
    target_labels = [alpha_char == item['answer'] for alpha_char in range(len(alphanumeric_options))]
    prediction_probabilities = (np.float64(selection_log_prob_opt_option)).tolist()

    result = {
      "selection_results": selection_results ,
      "chosen": selections[chosen_selection],
      "answer": selections[item['answer']] 
    }
    return (result , prediction_probabilities, target_labels)

# TODO do chunked writing.    
def run_inference(model, tokenizer, dataset,
                  tag="default_tag", include_prompt=False, 
                  verbose = False, 
                  n_shots=1):
  results = []
  prediction_probabilities = []
  target_labels = []

  for question_idx, item  in enumerate(dataset):
    prompt, selections = generate_n_shot_prompt(dataset, question_idx, n_shots=n_shots)
    result, single_prediction_probabilities, single_target_labels = \
      run_single_inference(model, tokenizer, prompt, selections, 
                           item, verbose=verbose)
    target_labels += single_target_labels
    prediction_probabilities += single_prediction_probabilities 
    if tag:
      result["model_tag"] = tag
    if include_prompt:
      result["prompt_template"] = prompt 
    # save iteration.
    results.append(result)
    # save output file in chunks.
  return results, prediction_probabilities, target_labels