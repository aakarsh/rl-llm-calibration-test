import torch

from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from datasets import (
    load_dataset,
    Dataset
)

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
        # Maybe normalize tokent length.
        completion_log_prob = torch.mean(
                continuationConditionalLogProbs
        ).cpu()

        return completion_log_prob
