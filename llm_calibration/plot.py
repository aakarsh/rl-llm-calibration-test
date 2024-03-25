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
# for each bin you will have the probability of the points which land in the bin. 
# and the labels, , then you normalized frequency of the labels 1 for the y coordinate. 
#   
# A -(.9,0), B - 1 , C - 0  D 0 
def plot_calibration(prediction_probabilities, actual_labels,num_bins=50, range_start = 0 , range_end=.00001, out_file=None):
  """
  Plot the calibration curves for the predicted probabilities of a model.
  """
  sns.set_theme()
  # Sort predictions and corresponding actual labels
  sorted_indices = np.argsort(prediction_probabilities)
  sorted_probs = prediction_probabilities[sorted_indices]
  sorted_labels = actual_labels[sorted_indices]

  # Create equal-sized bins
  bin_edges = np.linspace(range_start, range_end, num_bins)
  # Calculate frequency of correct predictions in each bin
  bin_counts = np.zeros(num_bins)
  bin_correct = np.zeros(num_bins)

  for i in range(len(sorted_probs)):
      bin_index = np.digitize(sorted_probs[i], bin_edges) - 1
      if len(bin_counts) < bin_index:
          print("bin_index", bin_index)
      bin_counts[bin_index] += 1
      bin_correct[bin_index] += sorted_labels[i]
      
  # Find indices of zero counts
  bin_accuracy = np.zeros(num_bins)
  zero_counts_idx = np.where(bin_counts == 0)[0]

  # Set accuracy to 0 for bins with zero counts
  bin_accuracy[zero_counts_idx] = 0

  # Calculate accuracy for bins with non-zero counts
  bin_accuracy[bin_counts != 0] = bin_correct[bin_counts != 0] / bin_counts[bin_counts != 0]
  bin_means = (bin_edges[:-1] + bin_edges[1:]) / 2  # For plotting midpoints

  # Create the calibration chart
  plt.plot(bin_means, bin_means, '--', color='gray', label='Perfect Calibration')  # Diagonal line
  plt.plot(bin_means, bin_accuracy[:-1], 'o-', label='Model Calibration')
  plt.xlabel('Prediction Probability (P(True))')
  plt.ylabel('Frequency of Correct Predictions')
  plt.title('Calibration Chart')
  plt.legend()
  if out_file: plt.savefig(out_file)
  plt.show()
    
# Plot the calibration effect of the following
# convert boolean array to np.int32
