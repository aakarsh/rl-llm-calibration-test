#%%
import sys
import os

#%%
file_path = os.path.abspath(os.path.dirname(__file__))
report_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__))+"/../report/figures/")
 
sys.path.append(os.path.abspath(file_path+'/..'))
#%%
import importlib
import llm_calibration
importlib.reload(llm_calibration)
from llm_calibration.model.model_probability import get_normalized_probabilities
from llm_calibration.plot import generate_comparison_plot

#%% 
spyder_mode='IPYKERNEL_CELL_NAME' in os.environ

# Load the test data 
#%%  
                                
#%%   
# Generate the 0-shot calibration plot between 7-b, 14-b and 70-b models
# model_results_model_meta-llama_Llama-2-13b-hf_ds_all_tag-result.json
model_result_files = [
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_all_tag-result.json'), 
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_tag-result.json') 
]
generate_comparison_plot(model_result_files, 
                         ["Llama 13-b Base Model (0-shot)", 
                          "Llama 13-b Chat Model (0-shot)"
                          ], 
                         output_dir=report_path, 
                         output_tag="0-shot-13-b-chat-vs-13-b-hf")

#%%   
# Generate the 0-shot calibration plot between 7-b, 14-b and 70-b models
model_result_files = [
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_tag-result.json'), 
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_all_tag-result.json') 
]

generate_comparison_plot(model_result_files, 
                         ["Llama 13b Chat Model (0-shot)", 
                          "Llama 7b Chat Model (0-shot)"
                          ], 
                         output_dir=report_path, 
                         output_tag="0-shot-7b-vs-13b-chat")

# Generate the 5-shot calibration plot between 7-b, 14-b and 70-b models 
# generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="5-shot")

# Generate Subject-wise calibration plots between 7-b, 14-b models.
# generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="by_subject")