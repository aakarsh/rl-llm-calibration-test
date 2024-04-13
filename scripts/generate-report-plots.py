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
from llm_calibration.plot import generate_comparison_plot,generate_roc_plot

#%% 
spyder_mode='IPYKERNEL_CELL_NAME' in os.environ

# Load the test data 
#%%  
#%% MMLU 
#%% MMLU Zero Shot 
model_result_files = [
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-hf_ds_all_n_shots_1_tag-result.json'),
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_all_tag-result.json'),
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_tag-result.json'),
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_all_tag-result.json'),
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-70b-chat-hf_ds_all_n_shots_1_tag-result.json'),
   
]
generate_comparison_plot(model_result_files, 
                         [
                            "Llama 7-b Base Model (0-shot)",
                            "Llama 7-b Chat Model (0-shot)",
                            "Llama 13-b Base Model (0-shot)",
                            "Llama 13-b Chat Model (0-shot)",
                            "Llama 70-b Chat Model (0-shot)"
                          ],
                         dynamic_bins=True, 
                         samples_per_bin=1000,
                         output_dir=report_path, 
                         output_tag="0-shot-MMLU")

generate_roc_plot(model_result_files, 
                         [
                            "Llama 7-b Base Model (0-shot)",
                            "Llama 7-b Chat Model (0-shot)",
                            "Llama 13-b Base Model (0-shot)",
                            "Llama 13-b Chat Model (0-shot)",
                            "Llama 70-b Chat Model (0-shot)"
                          ],
                         output_dir=report_path, 
                         output_tag="0-shot-MMLU")

#%% MMLU 5-shot
model_result_files = [
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-hf_ds_all_n_shots_5_tag-result.json'),
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_all_n_shots_5_tag-result.json'),
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_all_n_shots_5_tag-result.json'),
   os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_all_n_shots_5_tag-result.json') 
]
generate_comparison_plot(model_result_files, 
                         [
                          "Llama 7-b Base Model - MMLU - Overall (5-shot)",
                          "Llama 7-b Chat Model - MMLU - Overall (5-shot)",
                          "Llama 13-b Base Model - MMLU Overall (5-shot)",
                          "Llama 13-b Chat Model - MMLU Overall (5-shot)"
                         ],
                         dynamic_bins=True, 
                         samples_per_bin=500,
                         output_dir=report_path, 
                         output_tag="5-shot-MMLU")

generate_roc_plot(model_result_files, [
                          "Llama 7-b Base Model (5-shot)",
                          "Llama 7-b Chat Model (5-shot)",
                          "Llama 13-b Base Model (5-shot)",
                          "Llama 13-b Chat Model (5-shot)"
                         ],
                         output_dir=report_path, 
                         output_tag="5-shot-MMLU")

#%% MMLU-Subject Specific:
#%% MMLU-Subject Specific 0-shot 
## TODO: 7b , 7b-chat, 13b
subjects = ["STEM", "SOCIAL_SCIENCE", "HUMANITIES", "OTHER"]
model_result_files = [ os.path.abspath(file_path+('/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_%s_tag-result.json'% subject)) for subject in subjects]
generate_comparison_plot(model_result_files, 
                          ["Llama 13-b Chat Model %s (0-shot)" % subject.upper() for subject in subjects], 
                         dynamic_bins=True, 
                         samples_per_bin=300,
                         output_dir=report_path, 
                         output_tag="0-shot-MMLU-subjects")

generate_roc_plot(model_result_files, 
                          ["Llama 13-b Chat Model %s (0-shot)" % subject.upper() for subject in subjects], 
                         output_dir=report_path, 
                         output_tag="0-shot-MMLU-subjects")


#%% MMLU-Subject Specific 5-shot
## TODO: 7b , 7b-chat, 13b
subjects = ["STEM", "SOCIAL_SCIENCE", "HUMANITIES", "OTHER"]
model_result_files = [os.path.abspath(file_path+('/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_%s_n_shots_5_tag-result.json'% subject)) 
                           for subject in subjects]
generate_comparison_plot(model_result_files, 
                          ["Llama 13-b Chat Model %s (5-shot)" % subject.upper() for subject in subjects], 
                         dynamic_bins=True, 
                         samples_per_bin=300,
                         output_dir=report_path, 
                         output_tag="5-shot-MMLU-subjects")

generate_roc_plot(model_result_files, 
                          ["Llama 13-b Chat Model %s (0-shot)" % subject.upper() for subject in subjects], 
                         output_dir=report_path, 
                         output_tag="5-shot-MMLU-subjects")
#%% LogicQA
#%% Logic-QA 0-shot
model_result_files = [
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_logic_qa_n_shots_1_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_logic_qa_n_shots_1_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-hf_ds_logic_qa_n_shots_1_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_logic_qa_n_shots_1_tag-result.json')
]

generate_comparison_plot(model_result_files, 
                           [
                           'Llama 13b Chat Model (0-shot)',
                           'Llama 13b Base Model (0-shot)',
                           'Llama 7b Base Model (0-shot)',
                           'Llama 7b Chat Model (0-shot)' 
                           ], 
                         dynamic_bins=True, 
                         samples_per_bin=300,
                         output_dir=report_path, 
                         output_tag="0-shot-logic-qa")

generate_roc_plot(model_result_files, 
                          [ 
                           'Llama 13b Chat Model (0-shot)',
                           'Llama 13b Base Model (0-shot)',
                           'Llama 7b Base Model (0-shot)',
                           'Llama 7b Chat Model (0-shot)' 
                           ], 
                         output_dir=report_path, 
                         output_tag="0-shot-logic-qa")

#%% Logic-QA 5-shot 
model_result_files = [
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_lucasmccabe_logiqa_n_shots_5_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-hf_ds_lucasmccabe_logiqa_n_shots_5_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_lucasmccabe_logiqa_n_shots_5_tag-result.json'), 
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_lucasmccabe_logiqa_n_shots_5_tag-result.json') 
]
generate_comparison_plot(model_result_files, 
                          [
                           'Llama 7b Chat Model (5-shot)', 
                           'Llama 7b Base Model (5-shot)', 
                           'Llama 13b Base Model (5-shot)',
                           'Llama 13b Chat Model (5-shot)'
                           ], 
                         dynamic_bins=True, 
                         samples_per_bin=200,
                         output_dir=report_path, 
                         output_tag="5-shot-logic-qa")

generate_roc_plot(model_result_files, 
                          [
                           'Llama 7b Chat Model (5-shot)', 
                           'Llama 7b Base Model (5-shot)', 
                           'Llama 13b Base Model (5-shot)',
                           'Llama 13b Chat Model (5-shot)'
                           ], 
                         output_dir=report_path, 
                         output_tag="5-shot-logic-qa")

#%% TruthQA 
#%% Truthful-QA 0-shot
model_result_files = [
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-hf_ds_truthful_qa_n_shots_1_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_truthful_qa_n_shots_1_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_truthful_qa_n_shots_1_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_truthful_qa_n_shots_1_tag-result.json')
]

generate_comparison_plot(model_result_files, [
                             'Llama 7b Base Model (0-shot)', 
                             'Llama 7b Chat Model (0-shot)', 
                              'Llama 13b Base Model (0-shot)', 
                              'Llama 13b Chat Model (0-shot)', 
                         ], 
                         dynamic_bins=True, 
                         samples_per_bin=100,
                         output_dir=report_path, 
                         output_tag="0-shot-truthful_qa")

generate_roc_plot(model_result_files, [
                             'Llama 7b Base Model (0-shot)', 
                             'Llama 7b Chat Model (0-shot)', 
                              'Llama 13b Base Model (0-shot)', 
                              'Llama 13b Chat Model (0-shot)', 
                         ], 
                         output_dir=report_path, 
                         output_tag="0-shot-truthful_qa-roc")

#%% Truthful-QA 5-shot
model_result_files = [
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_truthful_qa_n_shots_5_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-hf_ds_truthful_qa_n_shots_5_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-chat-hf_ds_truthful_qa_n_shots_5_tag-result.json'),
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-13b-hf_ds_truthful_qa_n_shots_5_tag-result.json')
]
generate_comparison_plot(model_result_files, 
                          [
                             'Llama 7b Chat Model (5-shot)', 
                             'Llama 7b Base (5-shot)',
                             'Llama 13b Chat Model (5-shot)', 
                             'Llama 13b  Base Model (5-shot)', 
                           ], 
                         dynamic_bins=True, 
                         samples_per_bin=100,
                         output_dir=report_path, 
                         output_tag="5-shot-TruthQA")

generate_roc_plot(model_result_files, 
                           [
                             'Llama 7b Chat Model (5-shot)', 
                             'Llama 7b Base (5-shot)',
                             'Llama 13b Chat Model (5-shot)', 
                             'Llama 13b  Base Model (5-shot)', 
                           ],
                         output_dir=report_path, 
                         output_tag="5-shot-TruthfulQA")


#### Experimental
#%% Running on human eval
model_result_files = [
    os.path.abspath(file_path+'/../output/model-output/model_results_model_meta-llama_Llama-2-7b-chat-hf_ds_openai_humaneval_n_shots_5_tag-result.json')
]
generate_comparison_plot(model_result_files,['Llama 7b Chat Model : Human Eval(0-shot)'],
                         dynamic_bins=True, 
                         samples_per_bin=25,
                         output_dir=report_path,
                         output_tag='0-shot-7b-human-eval')

#%% illanca  Human eval results:
model_result_files = [ 
    os.path.abspath(file_path+'/../output/model-output/'+model_result_file)
                      for model_result_file in [ "humaneval_7B_3options_final-_none_chat.json"
                                                ,"humaneval_7B_3options_final-chat.json"
                                                ,"humaneval_7B_3options_final.json"
                                                ,"humaneval_7B_3options_none_final.json"]]

generate_comparison_plot(model_result_files,
                         [
                              #"humaneval_7B_3options_final-_none_chat.json"
                              "humaneval_7B_3options_final-chat.json"
                              #,"humaneval_7B_3options_final.json"
                              #,"humaneval_7B_3options_none_final.json" 
                            ],
                         dynamic_bins=True, 
                         samples_per_bin=25,
                         output_dir=report_path,
                         output_tag='0-shot-7b-human-eval')

