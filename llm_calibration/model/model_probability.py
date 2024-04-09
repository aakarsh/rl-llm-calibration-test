import numpy as np
import torch

def get_normalized_probabilities(model_results, include_true_negatives=False):
  """
  Takes model results and returns normalized probabilities and truth values.
  """
  completions = list(sorted(model_results[0]['selection_results'].keys()))
  completion_probabilities = []
  predicted_probability = []
  truth_value = []
  actual_labels = []
  correct_predictions = []
  for model_result in model_results:
    model_log_prob_of_completion = torch.tensor([model_result['selection_results'][completion] for completion in completions])
    if model_log_prob_of_completion.isnan().any():
            print('Skipping NaN in model results.')
            continue
    model_completion_probability = torch.nn.functional.softmax(model_log_prob_of_completion, dim=0)
    # convert to probability 
    completion_probabilities += model_completion_probability.tolist()
    # The truth value is when the model answer is also the CORRECT answer. 
    # Question: But then are the model unchosen correct answers also correct ?
    # Question: Should we use a ROC curve to evaluate the model ?
    # Question: Should we consider the KL-divergence of model predictions ?
    # Question: We get swamped by true negatives, thus we don't include 
    # them unless explicitly required to. 
    for completion_idx, completion in enumerate(completions):
        if completion == model_result['chosen']:
                # Correct answer: TP.
                if completion == model_result['answer']: 
                        truth_value.append(1) 
                        actual_labels.append(1)
                        correct_predictions.append(1)
                        predicted_probability.append(model_completion_probability[completion_idx].item())
                else: # Incorrect answer: FN.
                        predicted_probability.append(model_completion_probability[completion_idx].item())
                        correct_predictions.append(0)
                        actual_labels.append(0)
                        truth_value.append(0) 
        else: # Completion was not chosen.
                # Incorrect answer: FN.
                if completion == model_result['answer']:
                        predicted_probability.append(model_completion_probability[completion_idx].item())
                        correct_predictions.append(0)
                        actual_labels.append(1)
                        truth_value.append(0)
                # Correct Answer: True Negative
                elif include_true_negatives:
                        """ 
                        As we have a abundance of true negatives, 
                        treating them as correct predictions leads to 
                        noisy results. Thus we don't include them, unless
                        explicitly required to.
                        """
                        predicted_probability.append(model_completion_probability[completion_idx].item())
                        correct_predictions.append(1)
                        actual_labels.append(0)
                        truth_value.append(0)
  return predicted_probability, \
          truth_value, \
          actual_labels, \
          correct_predictions,\
          completions

def summarize_model_results(model_results):
        completion_probabilities, truth_values, actual_values, predicted_probabilities, completions = get_normalized_probabilities(model_results)
        correct_predictions = np.sum(truth_values)  
        print("Correct Predictions", correct_predictions)
        print("Incorrect Predictions", len(truth_values) - correct_predictions)
        for i in range(len(completion_probabilities), len(completions)):
                for j in range(len(completions)):
                        print("Completion: ", completions[j], "Probability: ", completion_probabilities[i+j], "Truth Value: ", truth_values[i+j])
                        
def get_log_prob_of_completion(
        model,
        tokenizer,
        prompt,
        completion,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
):
        """
        Convenience function for computing the log probability of a completion
        given a prompt.
        """
        # tokenize the prompt and the completion
        # truncate so as to fit into to maximal context window of gpt-2
        # which is 1024 tokens
        input_ids = tokenizer(
                prompt + completion,
                return_tensors='pt',
                truncation=True,
                max_length=1024,
        )['input_ids'].to(device)

        # create attention mask and position ids
        attention_mask = (input_ids != tokenizer.eos_token_id).to(dtype=torch.int64)
        position_ids = attention_mask.cumsum(-1)-1
        # get the logits for the completion
        with torch.no_grad():
                out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids
                )

        # get the logits of the completion
        # for that, make a tensor of the logits
        # for the completion only
        # in particular, we shift the indices by one to
        # the left to access logits of the
        # actual sequence tokens
        logits_completion = out.logits[:, :-1]
        logits_completion = logits_completion.squeeze()
        # get the log probabilities for the completion
        log_probs = torch.nn.functional.log_softmax(
                logits_completion,
                dim=-1
        )
        # retrieve the logit corresponding to the actual completion tokens
        try:
                log_completion_tokens = log_probs.gather(
                        dim=-1,
                        index=input_ids[:, 1:].squeeze().unsqueeze(-1)
                )
        except:
                log_completion_tokens = log_probs.gather(
                        dim=-1,
                        index=input_ids[:, 1:].unsqueeze(-1)
                )

        # separately tokenize prompt
        # so as to access the logits for the completion only
        # when scoring the completion
        input_ids_prompt = tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=1024
        )['input_ids'].to(device)
        prompt_end_index = input_ids_prompt.shape[-1] - 1

        continuationConditionalLogProbs = log_completion_tokens[
            prompt_end_index:
        ]
        # Why are we doing a mean here ? 
        # Normalizing over number of token, because when we have 
        # log probablity of different completions, 
        # because under some tokenizations we might have 
        # different number tokens, when we have sum of probablity of two tokens, 
        # you might get a smaller number than so to avoid the issue of 
        # effect of number tokens, to take the average over the number ot tokens, 
        # so we compute the average token probability, if they consist only of one token or two tokens, you
        # can compute the number
        # We can do that when we have variation in token number, 
        # sum is totally fine. 
        
        # Maybe normalize tokent length.
        completion_log_prob = torch.mean(
                continuationConditionalLogProbs
        ).cpu()

        return completion_log_prob
