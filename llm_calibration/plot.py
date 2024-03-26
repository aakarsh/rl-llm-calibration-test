#%%
import seaborn as sns
import json
import os
from .model.model_probability import get_normalized_probabilities
import numpy as np
import matplotlib.pyplot as plt

#%%
# Get probability for each to option. 
# 0-1  - 10  
# For each bin you will have the probability of the points which land in the bin. 
# and the labels, , then you normalized frequency of the labels 1 for the y coordinate. 
#   
# A -(.9,0), B - 1 , C - 0  D 0 
def plot_calibration(prediction_probabilities, actual_labels,
                     num_bins=10, range_start = 0, range_end=1, 
                     model_label="Model Calibration",
                     out_file=None, figure=None, show_figure=False):
  """
    Plot the calibration curves for the predicted probabilities of a model.
  """
  sns.set_theme()

  # Sort predictions and corresponding actual labels
  sorted_indices = np.argsort(prediction_probabilities)
  prediction_probabilities = np.array(prediction_probabilities)
  actual_labels = np.array(actual_labels)
  sorted_probs = prediction_probabilities[sorted_indices.astype(int)]
  sorted_labels = actual_labels[sorted_indices.astype(int)]

  # Create equal-sized bins
  bin_edges = np.linspace(range_start, range_end, num_bins)
  # Calculate frequency of correct predictions in each bin
  bin_counts = np.zeros(num_bins)
  bin_correct = np.zeros(num_bins)

  for i in range(len(sorted_probs)):
      bin_index = np.digitize(sorted_probs[i], bin_edges) - 1
      if len(bin_counts) < bin_index:
          pass
          #print("bin_index", bin_index)
      bin_counts[bin_index] += 1
      bin_correct[bin_index] += sorted_labels[i]
      
  # Find indices of zero counts
  bin_accuracy = np.zeros(num_bins)
  zero_counts_idx = np.where(bin_counts == 0)[0]

  # Set accuracy to 0 for bins with zero counts
  bin_accuracy[zero_counts_idx] = np.nan

  # Calculate accuracy for bins with non-zero counts
  bin_accuracy[bin_counts != 0] = bin_correct[bin_counts != 0] / bin_counts[bin_counts != 0]
  bin_means = (bin_edges[:-1] + bin_edges[1:]) / 2  # For plotting midpoints

  mask = np.isfinite(bin_accuracy)
  mask = mask[:-1] # Last bin is not included in mask
  # Create the calibration chart
  if not figure: 
    figure = plt.figure(figsize=(10, 7))
    
  plt.plot(bin_means, bin_means, '--', color='gray', label='Perfect Calibration')  # Diagonal line
  plt.plot(bin_means[mask], bin_accuracy[:-1][mask], 'o-', label=model_label)
  plt.xlabel('Prediction Probability (P(True))')
  plt.ylabel('Frequency of Correct Predictions')
  plt.title('Calibration Chart')
  plt.legend()

  if out_file: 
    plt.savefig(out_file)
  if show_figure: 
    plt.show()
  
  return figure

#%%
def plot_calibration_comparison(model_tags, model_labels, 
                                prediction_probabilities, 
                                actual_labels, 
                                num_bins=10, range_start = 0 , range_end=1, out_file=None, show_figure=False):
  """
  Plot the calibration comparison plot.
  """
  save_fig = plt.figure(figsize=(10, 7))
  for idx, model_key in enumerate(model_tags):
    probabilities = prediction_probabilities[model_key]
    truth_values = actual_labels[model_key] 
    assert len(probabilities) == len(truth_values)
    plot_calibration(probabilities, truth_values, 
                     num_bins, range_start, range_end,
                     model_label=model_labels[idx], 
                     figure=save_fig, 
                     show_figure=False) 
  if out_file: 
    save_fig.savefig(out_file)
  if show_figure: 
    save_fig.show()
  return save_fig

def generate_comparison_plot(file_paths,  model_labels=[], 
                            output_dir=None, output_tag=None):
  """
  Generate a comparison on multiple runs of a model, 
  by parsing model result files.
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

def generate_calibration_plot(file_path, output_dir=None, output_tag=None):
  with open(file_path) as f:
    test_data = json.load(f)
  model_results = test_data[0]['results']
  completion_probabilities, truth_values = get_normalized_probabilities(model_results)
  assert len(completion_probabilities) == len(truth_values)
  
  plot_calibration(np.array(completion_probabilities), 
                  np.array(truth_values, dtype=np.int32), 
                  num_bins=10, range_start=0, range_end=1, out_file=output_dir+"/"+output_tag+".png")
