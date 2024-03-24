import llm_calibration.model.model_probability as mp
import numpy as np

model_results =  [{
            "model": "llama",
            "prompt_template": "Select one (A, B, C, D). Question: This question refers to the following information.\nNo task is more urgent than that of preserving peace. Without peace our independence means little. The rehabilitation and upbuilding of our countries will have little meaning. Our revolutions will not be allowed to run their course. What can we do? We can do much! We can inject the voice of reason into world affairs. We can mobilize all the spiritual, all the moral, all the political strength of Asia and Africa on the side of peace. Yes, we! We, the peoples of Asia and Africa, 1.4 billion strong.\nIndonesian leader Sukarno, keynote address to the Bandung Conference, 1955\nThe passage above is most associated with which of the following developments?\n\nA. The formation of the non-aligned movement\nB. Global disarmanent and nuclear non-proliferation\nC. The Green Revolution in agriculture\nD. Mobilization of pan-Asian ideology",
            "context_results": {
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
    assert np.sum(completion_probabilities) == 1