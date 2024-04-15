#%%
import seaborn as sns
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os
import re


import itertools
import functools

import sklearn.metrics as sk_metrics
from sklearn.metrics import RocCurveDisplay, roc_curve


#%%
from llm_calibration.model.model_probability import (get_normalized_probabilities, 
                                                     compute_ece,
                                                     compute_rms_calibration_error,
                                                     bin_prediction_probabilities_by_samples_per_bin)

def add_calibration_line_to_plot(calibration_line_num_points  = 1000):
    calibration_line_num_points  = 1000 
    y_space = np.linspace(0, 1, calibration_line_num_points)
    x_space = np.linspace(0, 1, calibration_line_num_points)
    plt.plot(x_space, y_space, '--', color='gray', label='Perfect Calibration')  
  
#%% 
def plot_calibration_equally_weighted_bins(prediction_probabilities, 
                                           actual_labels,
                                           samples_per_bin=100, 
                                           range_start=0, 
                                           range_end=1, 
                                           show_calibration_line=False,
                                           model_label="Model Calibration", 
                                           out_file=None, 
                                           figure=None, 
                                           show_figure=False):
  """
  Each bin will have an equal number of samples, given by the 
  samples_per_bin parameter. Thus it would be expected that regions 
  of the plot will have more samples and bins than others.
  """
  sns.set_theme();
  
  bin_accuracy,bin_mean_probability, _=\
    bin_prediction_probabilities_by_samples_per_bin(prediction_probabilities, actual_labels, samples_per_bin)
 
  if not figure:
    figure = plt.figure(figsize=(10, 10))

  if show_calibration_line:   
    add_calibration_line_to_plot()

  plt.plot(bin_mean_probability, bin_accuracy, 'o-', label=model_label)

  plt.xlim(0,1)
  plt.ylim(0,1)

  plt.xlabel('Prediction Probability (P(True))')
  plt.ylabel('Frequency of Correct Predictions')
  plt.title('Calibration Chart')
  plt.legend()

  if out_file: 
    plt.savefig(out_file)
  if show_figure: 
    plt.show()
  
  return figure;


#%%
def plot_calibration(prediction_probabilities, actual_labels,
                     num_bins=10, range_start = 0, range_end=1, 
                     model_label="Model Calibration",
                     out_file=None, figure=None, show_figure=False):
  """
    Takes the probability for each completion along with its actual label(True/False), 
    and bins the probabilities into num_bins, and calculates the frequency of the
    correct predictions in each bin. 
    
    Plots the prediction probability against the frequency of correct predictions.  
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
                                dynamic_bins=True, 
                                samples_per_bin=100,
                                num_bins=10, 
                                range_start = 0 , 
                                range_end=1,
                                model_ece={},
                                model_rms_error={},
                                out_file=None, 
                                show_figure=False):
  """
  Plot the calibration comparison plot.
  """
  save_fig = plt.figure(figsize=(10, 7))
  for idx, model_key in enumerate(model_tags):
    print(model_key)
    print(prediction_probabilities.keys())
    probabilities = prediction_probabilities[model_key]
    truth_values = actual_labels[model_key] 
    assert len(probabilities) == len(truth_values)
    if dynamic_bins:
      #probabilities = probabilities[:100]
      #truth_values = truth_values[:100]
      show_calibration_line = idx == 0
      current_model_label = model_labels[idx] 
      print("model_key", model_key) 
      print("model_ece", model_ece) 
      print("model_rms_error", model_rms_error) 
      print("model_key in model_ece", model_key in model_ece)
      print("model_key in model_rms_error)", model_key in model_rms_error)
      
      if (model_key in model_ece) and (model_key in model_rms_error):
        ece_error = model_ece[model_key]
        rms_error = model_rms_error[model_key]
        current_model_label += f' ECE: {ece_error:.2f}' + f' RMSE: {rms_error:.2f}'
        
      plot_calibration_equally_weighted_bins(probabilities, truth_values, 
                                             samples_per_bin=samples_per_bin, 
                                             range_start=0, range_end=1, 
                                             show_calibration_line=show_calibration_line,
                                             model_label=current_model_label, 
                                             figure=save_fig, 
                                             show_figure=False)
    else:
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


def has_glob_pattern(filename):
  """
  Checks if a filename contains a glob pattern using regular expressions.

  Args:
      filename: The filename to check.

  Returns:
      True if the filename contains a glob pattern, False otherwise.
  """
  pattern = r"[*?\[\]{}]"  # Matches glob pattern characters
  return bool(re.search(pattern, filename))

def merge_json_files(directory, pattern="*.json"):
  """
  Merges all JSON files matching the given pattern within a specified directory 
  into a single JSON object.

  Args:
      directory: The path to the directory containing the JSON files.
      pattern: A string representing the file pattern to search for (e.g., "*.json").

  Returns:
      A dictionary containing the merged data from all JSON files within the directory.
  """
  merged_data =[] 
  for filename in glob.glob(os.path.join(directory, pattern)):
    with open(filename, "r") as f:
      try:
        data = json.load(f)
        if len(data)>0:
          merged_data+=data  # Update the merged dictionary
      except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {filename}: {e}")
  return merged_data


def generate_comparison_plot(file_paths,  model_labels=[], 
                             dynamic_bins=True,
                             samples_per_bin=100,
                            output_dir=None, output_tag=None):
  """
  Generate a comparison on multiple runs of a model, 
  by parsing model result files.
  """
  comparison_files = []
  
  for file_path in file_paths: 
      if has_glob_pattern(os.path.basename(file_path)):
        merged_files = merge_json_files(os.path.dirname(file_path), os.path.basename(file_path)) 
        comparison_files.append(merged_files)
      else:
        with open(file_path) as f:
            comparison_files.append(json.load(f))
        
  model_completion_probabilities={}
  model_truth_values = {}
  model_ece = {}
  model_rms_error = {} 
  for idx, comparison_file in enumerate(comparison_files):
      model_results = comparison_file 
      completion_probabilities, _, actual_labels, correct_predictions, completions= \
          get_normalized_probabilities(model_results)

      ece_error = compute_ece(completion_probabilities,actual_labels, correct_predictions)# , samples_per_bin=samples_per_bin)
      rms_error = compute_rms_calibration_error(completion_probabilities,actual_labels, correct_predictions)#,samples_per_bin = samples_per_bin)

      current_label: str  = model_labels[idx] 
      model_rms_error[current_label] = rms_error
      model_ece[current_label] = ece_error
      
      model_completion_probabilities[current_label] = \
          completion_probabilities
      model_truth_values[current_label] = correct_predictions #actual_labels # truth_values
      
  assert len(completion_probabilities) == len(correct_predictions) #len(actual_labels)
  print("model_ece:", model_ece)
  print("model_rms_error:", model_rms_error)
  
  plot_calibration_comparison(model_labels, model_labels, 
                              model_completion_probabilities,
                              model_truth_values,
                              dynamic_bins=dynamic_bins,
                              samples_per_bin=samples_per_bin,
                              range_start=0, range_end=1, 
                              model_ece=model_ece,
                              model_rms_error=model_rms_error,
                              out_file=output_dir+"/"+output_tag+".png")

def generate_roc_plot(file_paths, model_labels=[], 
                        output_dir=None, 
                        output_tag=None,
                        show_figure=True,
                        figure=None):
  """
  Plot the classification ROC
  """
  comparison_files = []

  for file_path in file_paths: 
      if has_glob_pattern(os.path.basename(file_path)):
        merged_files = merge_json_files(os.path.dirname(file_path), os.path.basename(file_path)) 
        comparison_files.append(merged_files)
      else:
        with open(file_path) as f:
            comparison_files.append(json.load(f))
      
  model_completion_probabilities={}
  model_truth_values = {}

  if not figure:
    figure = plt.figure(figsize=(10, 10))
 
  for idx, comparison_file in enumerate(comparison_files):
      model_results = comparison_file 
 
      completion_probabilities, truth_value, actual_labels, correct_predictions, completions= \
          get_normalized_probabilities(model_results, include_true_negatives=True)
      current_label: str  = model_labels[idx]
      model_completion_probabilities[current_label] = \
          completion_probabilities
      model_truth_values[current_label] = correct_predictions #actual_labels # truth_values
      fpr, tpr, thresholds = sk_metrics.roc_curve(actual_labels, completion_probabilities)  
      roc_display = RocCurveDisplay.from_predictions(actual_labels, completion_probabilities, name=model_labels[idx], ax = plt.gca())
      
  chance_points = np.linspace(0, 1, 300) 
  plt.plot(chance_points,chance_points, linestyle='--', color='gray', scalex=False, scaley=False, label='Chance')
  plt.legend()

  if output_dir: 
    out_file=output_dir+"/"+output_tag+"-roc.png" 
    plt.savefig(out_file)
  if show_figure: 
    plt.show()
  
  plt.show() 
  return None


def generate_calibration_plot(file_path, output_dir=None, output_tag=None):
  with open(file_path) as f:
    test_data = json.load(f)
  model_results = test_data[0]['results']
  completion_probabilities, _, _, correct_predictions, _ = get_normalized_probabilities(model_results)
  assert len(completion_probabilities) == len(correct_predictions)
  
  plot_calibration(np.array(completion_probabilities), 
                  np.array(correct_predictions, dtype=np.int32), 
                  num_bins=10, range_start=0, range_end=1, out_file=output_dir+"/"+output_tag+".png")

# %%
