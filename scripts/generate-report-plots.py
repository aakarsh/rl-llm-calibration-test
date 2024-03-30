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
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_tag-result.json'),
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-hf_ds_all_n_shots_5_tag-result.json') ,
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_all_n_shots_5_tag-result.json') 
]
generate_comparison_plot(model_result_files, 
                         ["Llama 13-b Base Model (0-shot)", 
                          "Llama 13-b Chat Model (0-shot)",
                          "Llama 7-b Base Model (5-shot)",
                          "Llama 13-b Chat Model (5-shot)"
                          ], 
                         output_dir=report_path, 
                         output_tag="0-shot-13-b-chat-vs-13-b-hf")

#%%   
# Generate the 0-shot calibration plot between 7-b, 14-b and 70-b models
model_result_files = [
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_tag-result.json'), 
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_n_shots_5_tag-result.json'), 
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_all_tag-result.json'), 
]

generate_comparison_plot(model_result_files, 
                         [ "Llama 13b Chat Model (0-shot)", 
                           "Llama 13b Chat Model (5-shot)", 
                           "Llama 7b Chat Model (0-shot)",
                           "Llama 7b Chat Model (5-shot)"], 
                         output_dir=report_path, 
                         output_tag="0-shot-7b-vs-13b-chat")

#%% 0-shot vs 5-shot 
# 7-b can be better calibrated with smart prompting to bring calibration closer to 13b model.
model_result_files = [
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_tag-result.json'), 
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_n_shots_5_tag-result.json'), 
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_all_n_shots_5_tag-result.json') 
]

generate_comparison_plot(model_result_files, 
                         [
                            "Llama 13b Chat Model (0-shot)", 
                            "Llama 13b Chat Model (5-shot)",
                            "Llama 7b Chat Model (5-shot)" 
                          ], 
                         output_dir=report_path, 
                         output_tag="0-shot-vs-5-shot-7b-vs-13b-chat")

# Generate the 5-shot calibration plot between 7-b, 14-b and 70-b models 
# generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="5-shot")

# Generate Subject-wise calibration plots between 7-b, 14-b models.
# generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="by_subject")
#%%
subjects = ["STEM", "SOCIAL_SCIENCE", "HUMANITIES", "OTHER"]
model_result_files = [ os.path.abspath(file_path+('/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_%s_tag-result.json'% subject)) for subject in subjects]

generate_comparison_plot(model_result_files, 
                          ["Llama 13-b Chat Model %s (0-shot)" % subject.upper() for subject in subjects], 
                         output_dir=report_path, 
                         output_tag="0-shot-13-b-chat-vs-subjects")
#%% 5-shot subject specific.
subjects = ["STEM", "SOCIAL_SCIENCE", "HUMANITIES", "OTHER"]
model_result_files = [os.path.abspath(file_path+('/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_%s_n_shots_5_tag-result.json'% subject)) 
                           for subject in subjects]
generate_comparison_plot(model_result_files, 
                          ["Llama 13-b Chat Model %s (5-shot)" % subject.upper() for subject in subjects], 
                         output_dir=report_path, 
                         output_tag="5-shot-13-b-chat-vs-subjects")
#%% 5-shot logic-qa 
model_result_files = [
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_lucasmccabe_logiqa_n_shots_5_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-hf_ds_lucasmccabe_logiqa_n_shots_5_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_lucasmccabe_logiqa_n_shots_5_tag-result.json') 
]
generate_comparison_plot(model_result_files, 
                          ['Llama 7b Chat Model (5-shot)', 
                           'Llama 7b Base Model (5-shot)', 
                           'Llama 13b Chat Model (5-shot)'], 
                         output_dir=report_path, 
                         output_tag="5-shot-7b-13b-logic-qa")

#%% Running on human eval
model_result_files = [
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_openai_humaneval_n_shots_5_tag-result.json')
]
generate_comparison_plot(model_result_files,['Llama 7b Chat Model : Human Eval(0-shot)']
                         ,output_dir=report_path
                         ,output_tag='0-shot-7b-human-eval')
