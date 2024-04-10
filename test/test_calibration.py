import os
import json
import numpy as np
import scipy 

import llm_calibration.model.model_probability as mp
import llm_calibration.plot as plot
import llm_calibration.runner.logic_qa as logic_qa_runner
import llm_calibration.runner.mmlu as mmlu_runner
import llm_calibration.runner.multiple_choice_questions as mcq
import llm_calibration.runner.true_false_questions as tfq

file_path = os.path.abspath(os.path.dirname(__file__))

model_results =  [{
            "prompt_template": "Select one (A, B, C, D). Question: This question refers to the following information.\nNo task is more urgent than that of preserving peace. Without peace our independence means little. The rehabilitation and upbuilding of our countries will have little meaning. Our revolutions will not be allowed to run their course. What can we do? We can do much! We can inject the voice of reason into world affairs. We can mobilize all the spiritual, all the moral, all the political strength of Asia and Africa on the side of peace. Yes, we! We, the peoples of Asia and Africa, 1.4 billion strong.\nIndonesian leader Sukarno, keynote address to the Bandung Conference, 1955\nThe passage above is most associated with which of the following developments?\n\nA. The formation of the non-aligned movement\nB. Global disarmanent and nuclear non-proliferation\nC. The Green Revolution in agriculture\nD. Mobilization of pan-Asian ideology",
            "selection_results": {
                "A": 2.613432672199221e-06,
                "B": 1.4545943395145119e-06,
                "C": 5.854179790909773e-07,
                "D": 1.1065970186144855e-06
            },
            "chosen": "A",
            "answer": "A"
        }]

def load_model_results(file_path:str):
    model_results = None
    with open(file_path) as f:
            model_results = json.load(f)
    return model_results

def create_dummy_dataset():
    dummy_dataset=[]
    for i in range(10):
        ds = {"question":"What is question %d?" % i, 
              "answer": i % 4,
              "choices":["ans-1", "ans-2", "ans-3", "ans-4"]}
        dummy_dataset.append(ds)
    return dummy_dataset

def assert_valid_computed_normalized_probabilities(completion_probabilities, truth_values, actual_values, completions):
    """
    completion_probabilities - 
    truth_values - 
    actual_values - 
    completions -
    """
    for i in range(len(completion_probabilities), len(completions)):
        # Take a slice of the completion probabilities, truth values and actual values
        completion_probabilities = completion_probabilities[i:i+len(completions)]
        truth_values = truth_values[i:i+len(completions)]
        actual_values = actual_values[i:i+len(completions)]
        # The sum of the completion probabilities should be 1
        assert np.all(np.array(completion_probabilities) <= 1) and np.all(np.array(completion_probabilities) >= 0)
        # The number of truth values should be equal to the number of label values
        assert len(truth_values) == len(actual_values)
        # The sum of the truth values should be greater than the sum of the actual values, 
        # model can't have more correct answers
        assert np.sum(actual_values) >= np.sum(truth_values)


def assert_validate_model_result_file(file_path:str):   
    """
    Validate a model result file can be used to compute normalized probabilities. 
    """ 
    model_results = load_model_results(file_path)    
    completion_probabilities, truth_values, actual_values, predicted_probability, completions = \
        mp.get_normalized_probabilities(model_results)
    assert_valid_computed_normalized_probabilities(completion_probabilities, truth_values, actual_values, completions)
    # Try to 
    mp.summarize_model_results(model_results)
    mp.pretty_print_model_results(model_results)
        
def test_get_normalized_probabilities():
    """
    Given the model results.
    """
    completion_probabilities, truth_values, actual_values, predicted_probabilities, completions = mp.get_normalized_probabilities(model_results)
    
    assert_valid_computed_normalized_probabilities(completion_probabilities, truth_values, actual_values, completions)

def test_with_generated_model_result_file():
    """
    Test computation with actual file.
    """
    file_under_test = os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_all_tag-result.json') 
    assert_validate_model_result_file(file_under_test)
    mp.pretty_print_model_results(file_under_test)
    
def test_n_shot_prompt():
    dummy_dataset = create_dummy_dataset() 
    question_idx=1 
    generated_prompt, _ = mmlu_runner.generate_n_shot_prompt(
            dummy_dataset, 
            question_idx, 
            prompt_template="{question}", 
            options_template="({choice})", 
            n_shots=2)
    with open("test/data/expected_answer.txt") as f:
        expected_prompt = f.read().strip()
    assert generated_prompt == expected_prompt
    
def test_zero_shot():
   dummy_dataset = create_dummy_dataset() 
   question_idx=1
   generated_prompt, formatted_options = mmlu_runner.generate_n_shot_prompt(dummy_dataset, 
            question_idx, 
            n_shots=1)
   print(generated_prompt)
   with open("test/data/zero_shot.txt") as f:
        expected_prompt = f.read().strip()
   assert generated_prompt == expected_prompt
   assert formatted_options[0]  == "(A)"

def test_parse_item():
    item = {
        "context": "This is a context",
        "query": "This is a query",
        "options": ["A", "B", "C", "D"],
        "correct_option": 1
    }
    
    parsed_item = logic_qa_runner.parse_dataset_item(item)
    
    assert parsed_item["question"] == "This is a context\nQuestion:This is a query"
    assert parsed_item["choices"] == ["A", "B", "C", "D"]
    assert parsed_item["answer"] == 1
   
def create_logic_dummy_dataset():
    dummy_dataset=[]
    for i in range(10):
        ds = {'context': " Some Cantonese don't like chili, so some southerners don't like chili.",
                'query': 'Which of the following can guarantee the above argument?',
              'options': ['Some Cantonese love chili.',
                    'Some people who like peppers are southerners.',
                    'All Cantonese are southerners.',
                    'Some Cantonese like neither peppers nor sweets.'],
        'correct_option': i % 4}
        dummy_dataset.append(ds)
    return dummy_dataset
  
def test_zero_shot_logciqa():
   ldd = create_logic_dummy_dataset() 
   question = mcq.generate_n_shot_prompt(ldd, 1, n_shots=1, item_parser=logic_qa_runner.parse_dataset_item)
   print(question)
   
  
def create_completions_dataset():
    completions_dataset = []
    for i in range(10):
        ds = {"prompt": "This is prompt %d" % i, 
              "canonical_solution": "This is canonical solution %d" % i}
        completions_dataset.append(ds)
    return completions_dataset

def test_random_exclude():
    completions_dataset = create_completions_dataset()
    for i in range(10):
        selected_question_idx = 1
        alternate_index = tfq.random_index_excluding(completions_dataset, selected_question_idx)
        assert alternate_index != selected_question_idx
        assert alternate_index >= 0
        assert alternate_index < len(completions_dataset)
        
def test_make_boolean_question():
    completions_dataset = create_completions_dataset()
    print(completions_dataset)
    for i in range(10):
        selected_question_idx = 1
        question, options, actual_answer = tfq.make_single_boolean_question(selected_question_idx, completions_dataset)
        if actual_answer  == 0:
           assert "This is canonical solution %d"  % selected_question_idx in question
        print(question)
        
def test_bin_prediction_probabilities_by_samples_per_bin_perfect_calibration():
    """
    Given a perfect calibration model the bin accuracy should be 
    close to the bin mean probability. 
    """
    num_probabilities = 200000
    samples_per_bin = 1000 
    probabilities = np.random.rand(num_probabilities)
    actual_labels = [np.random.binomial(1, p) for p in probabilities]
    bin_accuracy, bin_mean_probability, bin_widths  = mp.bin_prediction_probabilities_by_samples_per_bin(probabilities, actual_labels, samples_per_bin=samples_per_bin)

    assert len(bin_accuracy) == len(bin_mean_probability) == (len(probabilities) // samples_per_bin)
    
    assert np.all(bin_accuracy >= 0) and np.all(bin_accuracy <= 1)
    assert np.all(bin_mean_probability >= 0) and np.all(bin_mean_probability <= 1)
    assert np.all(np.diff(bin_mean_probability) >= 0) # strictly increasing
    assert np.all(bin_widths > 0)
    # Assert that bin accuracy is close to bin mean probability, 
    # linear regression should be close to y = x
    result = scipy.stats.linregress(bin_accuracy, bin_mean_probability)
    assert result.slope > .96 and result.slope < 1.04
        