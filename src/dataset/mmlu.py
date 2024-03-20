#%%
from datasets import load_dataset
import numpy as np


def load_dataset(name):
  #mmlu_dataset = load_dataset("cais/mmlu","high_school_world_history")
  return load_dataset("cais/mmlu",name

def run_inference(model,tokenizer, dataset):
  model_results = []
  results = []
  mmlu_prediction_probabilities = []
  mmlu_target_labels = []
  verbose = False

  for idx, mmlu_item  in enumerate(mmlu_dataset['test']):
    #if idx> 10000: break;
    question_template = "Select one (A, B, C, D). Question: {question}".format(**mmlu_item)
    choices_template = "\n"+"\n".join(["%s. %s" % (alpha, choice) for alpha, choice in zip(['A', 'B', 'C', 'D'], mmlu_item['choices'])])
    prompt = "%s\n%s" %(question_template, choices_template)
    selections = ["A", "B", "C","D"]
    if verbose:
      print(prompt)
    selection_log_prob_opt_option = []
    for selection_idx, selection  in enumerate(selections):

      log_prob_opt_option = get_log_prob_of_completion(
          model=model,
          tokenizer=tokenizer,
          prompt=prompt,
          completion=selection)

      if verbose:
        print("selection",selection, "log_prob:",log_prob_opt_option)

      selection_log_prob_opt_option.append(np.float64(log_prob_opt_option.detach().numpy()))

    selection_results = (dict(zip(selections, np.float64(np.exp(selection_log_prob_opt_option)))))
    chosen_selection = np.argmax(selection_log_prob_opt_option)

    # MMLU target labels
    mmlu_target_labels =mmlu_target_labels +  [i == mmlu_item['answer'] for i in range(4)]
    # just append the probabilities
    mmlu_prediction_probabilities += (np.exp(selection_log_prob_opt_option)).tolist()

    results.append({"model": "llama", "prompt_template": prompt,
                    "context_results": selection_results ,
                    "chosen": selections[chosen_selection],
                    "answer": selections[mmlu_item['answer']]})
    model_results.append({"model_tag": "llama", "results": results})
  return model_results, mmlu_prediction_probabilities, mmlu_target_labels


for idx, results in enumerate(model_results):
  if idx > 200: break
  for result in results["results"]:
    print("Model: {model}".format(**result))
    print("Prompt:{prompt_template} \n {context_results} Choice:\n{chosen} \nAnswer:{answer}".format(**result))
