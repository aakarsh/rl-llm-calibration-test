#%%
import sys
import os

import seaborn as sns
import json
import os
import numpy as np
import matplotlib.pyplot as plt
#%%
file_path = os.path.abspath(os.path.dirname(__file__))
report_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__))+"/../report/figures/")
 
sys.path.append(os.path.abspath(file_path+'/..'))
#%%
import importlib
import llm_calibration
importlib.reload(llm_calibration)
from llm_calibration.model.model_probability import get_normalized_probabilities
from llm_calibration.plot import plot_calibration
from llm_calibration.plot import plot_calibration_comparison

#%% 
spyder_mode='IPYKERNEL_CELL_NAME' in os.environ

# Load the test data 
def generate_calibration_plot(file_path, output_dir=None, output_tag=None):
  with open(file_path) as f:
    test_data = json.load(f)
  model_results = test_data[0]['results']
  completion_probabilities, truth_values = get_normalized_probabilities(model_results)
  assert len(completion_probabilities) == len(truth_values)
  
  plot_calibration(np.array(completion_probabilities), 
                  np.array(truth_values, dtype=np.int32), 
                  num_bins=10, range_start=0, range_end=1, out_file=output_dir+"/"+output_tag+".png")

#%%  
def generate_comparison_plot(file_paths,  model_labels=[], 
                             output_dir=None, output_tag=None):
    """
    Generate a comparison on multiple runs of a model.
    """
    comparison_files = []
   
    for file_path in file_paths: 
        with open(file_path) as f:
            comparison_files.append(json.load(f))
        
    model_completion_probabilities={}
    model_truth_values = {}
    
    for idx, comparison_file in enumerate(comparison_files):
        model_results = comparison_file 
        completion_probabilities, truth_values = \
            get_normalized_probabilities(model_results)
        current_label: str  = model_labels[idx]
        model_completion_probabilities[current_label] = \
            completion_probabilities
        model_truth_values[current_label] = truth_values
       
    assert len(completion_probabilities) == len(truth_values)
    
    # model_tags, 
    # model_labels, 
    # prediction_probabilities, 
    # actual_labels, num_bins=10, 
    # range_start = 0 , 
    # range_end=1, 
    # out_file=None, 
    # show_figure=False
    plot_calibration_comparison(model_labels, model_labels, 
                                model_completion_probabilities,
                                model_truth_values,
                                range_start=0, range_end=1, 
                                out_file=output_dir+"/"+output_tag+".png")
                                
#%%   
# Generate the 0-shot calibration plot between 7-b, 14-b and 70-b models
# model_results_model_meta-llama_Llama-2-13b-hf_ds_all_tag-result.json
model_result_files = [
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_all_tag-result.json'), 
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_tag-result.json') 
]
generate_comparison_plot(model_result_files, 
                         ["Llama 13-b Base Model", 
                          "Llama 13-b Chat Model"
                          ], 
                         output_dir=report_path, 
                         output_tag="0-shot-13-b-chat-vs-13-b-hf")

#%%
#%%   
# Generate the 0-shot calibration plot between 7-b, 14-b and 70-b models
# model_results_model_meta-llama_Llama-2-13b-hf_ds_all_tag-result.json
model_result_files = [
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_tag-result.json'), 
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_all_tag-result.json') 
]
generate_comparison_plot(model_result_files, 
                         ["Llama 13b Chat Model", 
                          "Llama 7b Chat Model"
                          ], 
                         output_dir=report_path, 
                         output_tag="0-shot-7b-vs-13b-chat")


# Generate the 5-shot calibration plot between 7-b, 14-b and 70-b models 
# generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="5-shot")

# Generate Subject-wise calibration plots between 7-b, 14-b models.
# generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="by_subject")
#%%
