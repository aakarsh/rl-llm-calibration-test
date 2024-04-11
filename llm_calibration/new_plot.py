"""
!pip install matplotlib

"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import trivia_qa



def bin_results(probs, labels, nbins=20):
    """ 
    Bins results according to probability and returns bins.
    Bins are made up of (avg prob, frequency label is correct)
    and contain the same number of elements
    """
    probs = np.array(probs)
    labels = np.array(labels)
    bins = []
    isort = probs.argsort()
    probs = probs[isort]
    labels = labels[isort]
    nelem = len(probs) // nbins
    for i in range(nbins):
        start, end = i*nelem, (i+1)*nelem
        propavg = np.mean(probs[start:end])
        freq = np.sum(labels[start:end]) / (end-start)
        bins.append((propavg, freq))
    return bins
          
def calibration_chart(results_list, nbins=20, model_label='the results'):
    for results in results_list:
        bins = bin_results([res["prob"] for res in results],
                           [res["label"] for res in results],
                           nbins=nbins)
        probs, freqs = list(zip(*bins))
        plt.plot(probs, freqs, 'o-', label=model_label)
    plt.plot([(0,0),(1,1)], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Probabilities')
    plt.ylabel('Frequencies')
    plt.title('Calibration: {}'.format(model_label))
    plt.legend()
    plt.show()
    
    
