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
from llm_calibration.model.model_probability import get_normalized_probabilities
from llm_calibration.plot import plot_calibration

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
                  num_bins=20, range_start=0, range_end=1, out_file=output_dir+"/"+output_tag+".png")
 

# Generate the 0-shot calibration plot between 7-b, 14-b and 70-b models
generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="0-shot")

# Generate the 5-shot calibration plot between 7-b, 14-b and 70-b models 
generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="5-shot")

# Generate Subject-wise calibration plots between 7-b, 14-b models.
generate_calibration_plot(os.path.abspath(file_path+'/../test/data/test_data.json'), output_dir=report_path, output_tag="by_subject")