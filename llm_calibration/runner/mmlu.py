#%%
import datasets as hugging_face_datasets 
import numpy as np

from model.model_probability import  get_log_prob_of_completion

def load_dataset(name):
  return hugging_face_datasets.load_dataset("cais/mmlu",name)

def run_inference(model, tokenizer, dataset, 
                  tag="default_tag", include_prompt=False, 
                  verbose = False):
  results = []
  prediction_probabilities = []
  target_labels = []

  for _, item  in enumerate(dataset['test']):
    question_template = "{question}".format(**item)
    alphanumeric_options = ['A', 'B', 'C', 'D'] 
    choices_template = "\n"+"\n".join(["(%s) %s" % (alpha, choice) for alpha, choice in zip(alphanumeric_options, item['choices'])])
    prompt = "%s\n%s" % (question_template, choices_template)
    selections = ["(%s)" % choice for choice in alphanumeric_options]
    if verbose: 
      print(prompt)
    selection_log_prob_opt_option = []
    for _, selection  in enumerate(selections):
      log_prob_opt_option = get_log_prob_of_completion(
          model=model,
          tokenizer=tokenizer,
          prompt=prompt,
          completion=selection)

      if verbose:
        print("selection",selection, "log_prob:",log_prob_opt_option)

      selection_log_prob_opt_option.append(np.float64(log_prob_opt_option.detach().numpy()))

    selection_results = (dict(zip(selections, np.float64(selection_log_prob_opt_option))))
    chosen_selection = np.argmax(selection_log_prob_opt_option)

    target_labels = target_labels + [i == item['answer'] for i in range(len(alphanumeric_options))]
    # just append the probabilities
    prediction_probabilities += (np.float64(selection_log_prob_opt_option)).tolist()

    result = {
      "selection_results": selection_results ,
      "chosen": selections[chosen_selection],
      "answer": selections[item['answer']] 
    } 
    
    if tag:
      result["model_tag"] = tag

    if include_prompt:
      result["prompt_template"] = prompt 
    results.append(result)

  return results, prediction_probabilities, target_labels


