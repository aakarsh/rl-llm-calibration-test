import datasets as hugging_face_datasets
import logging
import numpy as np
import json
from llm_calibration.model.model_probability import  get_log_prob_of_completion

logger = logging.getLogger(__name__)

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
    alphanumeric_options = [chr(item).upper() for item in range(ord("a"), ord("z") + 1)][0:len(item['choices'])]
    question_template = prompt_template.format(question=item["question"])
    choices_template = "\n"+"\n".join(["%s. %s" % (options_template.format(choice=alpha), choice) for alpha, choice in zip(alphanumeric_options, item['choices'])])
    question_prompt = "%s\n%s\n" % (question_template, choices_template)
    if with_answer:
      question_prompt += "\n"+ answer_template.format(answer=options_template.format(choice=alphanumeric_options[item['answer']]))+"\n"
    else:
      question_prompt += "\n"+ answer_template.format(answer="")
    return question_prompt
 
  item = item_parser(dataset[question_idx]) 
  alphanumeric_options = [chr(item).upper() for item in range(ord("a"), ord("z") + 1)][0:len(item['choices'])]

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
    if verbose: 
      logger.info(prompt)

    selection_log_prob_opt_option = []
    for _, selection  in enumerate(selections):
      # run inference here for each selection.
      log_prob_opt_option = get_log_prob_of_completion(model=model, tokenizer=tokenizer, 
                                                       prompt=prompt, completion=selection)
      if verbose:
        logger.info("selection",selection, "log_prob:",log_prob_opt_option)

      selection_log_prob_opt_option.append(np.float64(log_prob_opt_option.detach().numpy()))
    
    selection_results = (dict(zip(selections, np.float64(selection_log_prob_opt_option))))
    chosen_selection = np.argmax(selection_log_prob_opt_option)
    # Dataset specific
    alphanumeric_options = [chr(item).upper() for item in range(ord("a"), ord("z") + 1)][0:len(selection_results.keys())]
    #['A', 'B', 'C', 'D'] 
    target_labels = [alpha_char == item['answer'] for alpha_char in range(len(alphanumeric_options))]
    prediction_probabilities = (np.float64(selection_log_prob_opt_option)).tolist()

    if item['answer'] >= len(selections):
      print("getting out of bounds", item['answer'], selections)
    result = {
      "selection_results": selection_results ,
      "chosen": selections[chosen_selection],
      "answer": selections[item['answer']]  # getting out of bounds here
    }
    return (result , prediction_probabilities, target_labels)

def run_inference(model, tokenizer, dataset,
                  tag="default_tag", 
                  include_prompt=False, 
                  dataset_item_parser = lambda x: x,
                  alphanumeric_options = ['A', 'B', 'C', 'D'],
                  start_idx=0,
                  stop_idx=-1,
                  chunk_size=100,
                  write_chunks=True,
                  output_dir="",
                  verbose = False, 
                  n_shots=1):
  """
  Run model on entire dataset.
  """
  results = []
  prediction_probabilities = []
  target_labels = []
  chunk_start=start_idx
  chunk_stop=chunk_start+chunk_size
  for question_idx, item  in enumerate(dataset):
    if start_idx > 0 and question_idx < start_idx:
      continue
    if stop_idx > 0 and question_idx > stop_idx:
      break
    
    if question_idx % 100 == 0:
      logger.info("Processing question %d" % question_idx)
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
    # TODO: Write output in chunks.
    results.append(result)

    if write_chunks:
        if question_idx >= chunk_stop:
          print(tag, chunk_start, chunk_stop)
          output_file_name = "model_results_"+tag+"-result-chunk-"+str(chunk_start)+"-to-"+str(chunk_stop)+".json"
          output_file = output_dir+"/"+output_file_name
          with open(output_file, "w") as f:
            logger.info("Writing chunk to file: "+ str(output_file))
            json.dump(results[chunk_start:chunk_stop], f, indent=4)
          chunk_start = chunk_stop
          chunk_stop = chunk_start+chunk_size
  return results, prediction_probabilities, target_labels 
