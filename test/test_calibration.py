import llm_calibration.model.model_probability as mp
import numpy as np
import llm_calibration.runner.mmlu as mmlu_runner

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

def test_get_normalized_probabilities():
    completion_probabilities, _ = mp.get_normalized_probabilities(model_results)
    assert np.isclose(np.sum(completion_probabilities), 1)
    
def test_n_shot_prompt():
    dummy_dataset=[]
    for i in range(10):
        ds = {"question":"What is question %d?" % i, 
              "answer": i % 4,
              "choices":["ans-1", "ans-2", "ans-3", "ans-4"]}
        dummy_dataset.append(ds)
       
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