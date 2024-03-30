import random
import numpy as np

from llm_calibration.model.model_probability import get_log_prob_of_completion

boolean_question_template =\
    'Question : {question}.\n Proposed Answer: {answer}\n Is the following answer:  \n A) True \n B)False \n Proposed answer:'

def format_question(prompt):
    formatted = boolean_question_template.format(question = prompt["question"],
                                                 answer   = prompt["answer"])
    return formatted

def make_single_boolean_question(idx, completions_dataset):
    use_canonical_solution = bool(random.getrandbits(1))
    # randomly-select an index that is not the current-index.
    r = list(range(1,len(completions_dataset))) + list(range(idx+1, len(completions_dataset)))
    alternate_index = random.choice(r) 
    proposed_idx = idx if use_canonical_solution else alternate_index
    propose_solution = completions_dataset[proposed_idx]["canonical_solution"]
    prompt_dict = { "question": completions_dataset[idx]["prompt"],
                    "answer": propose_solution }
    formatted_options = [ "A", "B" ] 
    actual_answer =  0 if use_canonical_solution  else  1
    return format_question(prompt_dict),  formatted_options,  actual_answer

def run_inference(model, tokenizer, dataset,
                  tag="default_tag", include_prompt=False, 
                  dataset_item_parser = lambda x: x,
                  alphanumeric_options = ['A', 'B'],
                  verbose = False, 
                  n_shots=1):
  results = []
  prediction_probabilities = []
  target_labels = []

  for question_idx, item  in enumerate(dataset):
    if question_idx % 100 == 0:
      print("Processing question %d" % question_idx)
    item = dataset_item_parser(item)
    prompt, selections, actual_answer = make_single_boolean_question(question_idx, dataset)
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

    target_labels = target_labels + [i == actual_answer for i in range(len(alphanumeric_options))]
    # just append the probabilities
    prediction_probabilities += (np.float64(selection_log_prob_opt_option)).tolist()

    result = {
      "selection_results": selection_results ,
      "chosen": selections[chosen_selection],
      "answer": selections[actual_answer] 
    } 
    
    if tag:
      result["model_tag"] = tag

    if include_prompt:
      result["prompt_template"] = prompt 
    results.append(result)

  return results, prediction_probabilities, target_labels 

"""
def pick_questions(opennai_test):
    model_results = []
    results = []
    
    for idx, q in enumerate(opennai_test):
        if idx == 200:
            break
        print(idx)
        res = bool(random.getrandbits(1))
        selections = ["A","B"]
        if res == True:
            r = list(range(1,len(opennai_test))) + list(range(idx+1, len(opennai_test)))
            ind = random.choice(r)

            distractor = {"prompt":opennai_test[idx]["prompt"],
                          "canonical_solution": opennai_test[ind]["canonical_solution"]}
            disct_probs = format_question(distractor)
            disct_probs = np.asarray(disct_probs)
            selection_results = dict(zip(selections, disct_probs))
            chosen_selection = np.argmax(disct_probs)
            results.append({"model": "llama",
                        "context_results": selection_results ,
                        "chosen": selections[chosen_selection],
                        "answer": "B"})
            model_results.append({"model_tag": "llama", "results": results})

        else:
            act_probs = format_question(opennai_test[idx])
            disct_probs = np.asarray(act_probs)
            selection_results = dict(zip(selections, disct_probs))
            chosen_selection = np.argmax(disct_probs)
        results.append({"model": "llama",
                        "context_results": selection_results ,
                        "chosen": selections[chosen_selection],
                        "answer": "A"})
        model_results.append({"model_tag": "llama", "results": results})

    print(model_results)
"""
