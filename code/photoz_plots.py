import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from photoz_utils import *

def plot_comparison(metrics_array, legends, desc, zmax):
    """
    Pass in a list where the elements are the output get_point_metrics() from photoz_utils.py and compare some of the metrics on each plot.
    metrics_array: list
    legends: list
    desc: str
    zmax: float
    """
    index = len(metrics_array)
    sns.set(rc={'figure.figsize':(16,24), 'lines.markersize':10})
    plt.suptitle(f'{desc}')
    plt.subplots_adjust(hspace=0.2)
    
    metrics = ['bias_bw', 'scatter_bw', 'outlier_bw', 'loss', 'mse']
    for i in enumerate(metrics):
        plt.subplot(len(metrics), 1, i[0] + 1)
        for j in range(0,index):
            sns.lineplot(x=[0, zmax], y=[0, 0], linewidth=2, color='black')
            bin_lefts = metrics_array[j]['zspec_bin'].apply(lambda x: x.left)
            sns.scatterplot(x=bin_lefts, y=metrics_array[j][i[1]], marker = '.', edgecolor='none', label=f'{legends[j]}')
            plt.xlabel('Redshift')
    