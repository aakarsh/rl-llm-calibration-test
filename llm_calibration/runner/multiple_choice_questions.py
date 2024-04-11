import datasets as hugging_face_datasets
import numpy as np

from llm_calibration.model.model_probability import  get_log_prob_of_completion

def generate_n_shot_prompt(dataset,
                           question_idx,
                           prompt_template="{question}",
                           options_template="({choice})",
                           answer_template="Answer:{answer}",
                           item_parser=lambda x: x,
                           alphanumeric_options = ['A', 'B', 'C', 'D'],
                           n_shots=5):
  """
  Generate a n-shot prompt.
  """
  def format_question(current_idx, with_answer=True):
    item = item_parser(dataset[current_idx]) 
    question_template = prompt_template.format(question=item["question"])
    choices_template = "\n"+"\n".join(["%s. %s" % (options_template.format(choice=alpha), choice) for alpha, choice in zip(alphanumeric_options, item['choices'])])
    question_prompt = "%s\n%s\n" % (question_template, choices_template)
    if with_answer:
      question_prompt += "\n"+ answer_template.format(answer=options_template.format(choice=alphanumeric_options[item['answer']]))+"\n"
    else:
      question_prompt += "\n"+ answer_template.format(answer="")
    return question_prompt
 
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

def run_inference(model, tokenizer, dataset,
                  tag="default_tag", include_prompt=False, 
                  dataset_item_parser = lambda x: x,
                  alphanumeric_options = ['A', 'B', 'C', 'D'],
                  verbose = False, 
                  n_shots=1):
  """
  Run model on entire dataset.
  """
  results = []
  prediction_probabilities = []
  target_labels = []

  for question_idx, item  in enumerate(dataset):
    if question_idx % 100 == 0:
      print("Processing question %d" % question_idx)
    item = dataset_item_parser(item)
    prompt, selections = generate_n_shot_prompt(dataset, question_idx, 
                                                n_shots=n_shots, 
                                                item_parser=dataset_item_parser)
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
  return results, prediction_probabilities, target_labels 