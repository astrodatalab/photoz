import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from photoz_utils import *


def get_bias(true_labels, pred_labels, zmax):
    """
    Must also be using photoz_utils.py.
    :param true_labels: True redshift labels. Pandas Series
    :param pred_labels: Predicted redshift labels. Pandas Series
    :param zmax: Float of max redshift
    :return: Array of biases.
    """
    bins = pd.cut(true_labels, bins=np.linspace(0, zmax, 101)) # change when using different max redshift
    true_grouped = true_labels.groupby(bins)
    pred_grouped = pred_labels.groupby(bins)

    metrics_list = []
    for zspec_bin in true_grouped.groups:
        binned_z_true = true_grouped.get_group(zspec_bin)
        binned_z_pred = pred_grouped.get_group(zspec_bin)
        bias_bw = calculate_bias(binned_z_pred, binned_z_true)
        bias_conv = calculate_bias(binned_z_pred, binned_z_true, conventional=True)

        metrics_list.append([zspec_bin, bias_bw, bias_conv])

    metrics = pd.DataFrame(metrics_list, columns=['zspec_bin', 'bias_bw', 'bias_conv'])
    return metrics


def plot_bias(bias_array, name_array, desc, zmax):
    """
    Must also be using photoz_utils.py
    :param bias_array: List of bias arrays.
    :param name_array: List of strings which names each respective array in bias_array.
    :param desc: String that describe what is being compared.
    :param zmax: Float of max redshift
    :return: Plots bias arrays with given labels from name_array.
    """
    index = len(bias_array)
    sns.set(rc={'figure.figsize': (18, 8), 'lines.markersize': 20})
    sns.lineplot(x=[0, zmax], y=[0, 0], linewidth=2, color='black')
    plt.xlabel('Redshift')
    plt.ylabel(f'{desc}')

    for i in range(0, index):
        bin_lefts = bias_array[i]['zspec_bin'].apply(lambda x: x.left)
        sns.scatterplot(x=bin_lefts, y=bias_array[i]['bias_bw'], marker='.', edgecolor='white', label=name_array[i])
        #sns.scatterplot(x=bin_lefts, y=bias_array[i]['bias_conv'], marker = '.', facecolors='none', edgecolor='black')
