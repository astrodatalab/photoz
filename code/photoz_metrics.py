### Evaluating photo-z estimation given true redshifts.
### Metrics match up with those from https://arxiv.org/pdf/2003.01511.pdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import astropy.stats as astrostats
import scipy.stats as stats

def delz(z_photo, z_spec):
    """
    Vector of residuals.
    """
    dz = (z_photo-z_spec) / (1+z_spec)
    return dz


### POINT METRICS


def calculate_bias(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Returns a single value.
    """
    dz = delz(z_photo, z_spec)
    if (conventional):
        b = np.median(dz)
    else:
        b = astrostats.biweight_location(dz)
    return b


def calculate_scatter(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Returns a single value.
    """
    dz = delz(z_photo, z_spec)
    if (conventional):
        s = stats.median_abs_deviation(dz, scale='normal') # normal scale divides MAD by 0.67449
    else:
        s = np.sqrt(astrostats.biweight_midvariance(dz))
    return s


def calculate_outlier_rate(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Returns a single value.
    """
    dz = delz(z_photo, z_spec)
    if (conventional):
        outlier_scores = abs(dz)
        eta = np.mean(outlier_scores > 0.15)
    else:
        b = calculate_bias(z_photo, z_spec)
        s = calculate_scatter(z_photo, z_spec)
        outlier_scores = abs(dz - b)
        eta = np.mean(outlier_scores > (2*s))
    return eta


def calculate_loss(z_photo, z_spec):
    """
    HSC METRIC. Returns an array.
    """
    dz = delz(z_photo, z_spec)
    gamma = 0.15
    denominator = 1.0 + np.square(dz/gamma)
    L = 1 - 1.0 / denominator
    return L


### DENSITY METRICS


def calculate_PIT(z_photo_vectors, z_spec):
    z_spec = np.array(z_spec)
    z_photo_vectors = np.array(z_photo_vectors)
    length = len(z_photo_vectors[0])
    PIT  = np.zeros(len(z_photo_vectors))
    for i in range (len(z_photo_vectors)):          
        PIT[i] += len(np.where(z_photo_vectors[i]<z_spec[i])[0])*1.0/length
    return PIT


def calculate_CRPS(z_photo_vectors, z_spec):
    z_spec = np.array(z_spec)
    z_photo_vectors = np.array(z_photo_vectors)
    length = len(z_photo_vectors[0])
    crps = np.zeros(len(z_photo_vectors))
    for i in range(len(z_photo_vectors)):
        for j in range(200):
            z = 4.0*j/200
            if z < z_spec[i]:
                crps[i] += ((len(np.where(z_photo_vectors[i]<z)[0])*1.0/length)**2)*(4.0/200)
            else:
                crps[i] += ((len(np.where(z_photo_vectors[i]<z)[0])*1.0/length-1)**2)*(4.0/200)
    return crps


### ALL-METRICS FUNCTIONS


def get_point_metrics(z_photo, z_spec, binned=False):
    # create bins. if non-binned, treat entire set like a single bin.
    if (binned):
        bins = pd.cut(z_spec, bins=np.linspace(0, 4, 21))
    else:
        bins = pd.cut(z_spec, bins=np.linspace(0, 4, 2))
    true_grouped = z_spec.groupby(bins)
    pred_grouped = z_photo.groupby(bins)
    # calculate statistics for each bin
    metrics_list = []
    for zspec_bin in true_grouped.groups:
        # get redshifts of the bin
        binned_z_true = true_grouped.get_group(zspec_bin)
        binned_z_pred = pred_grouped.get_group(zspec_bin)
        # calculate statistics of the bin
        count = len(binned_z_true)
        L = np.mean(calculate_loss(binned_z_pred, binned_z_true))
        # biweights
        bias_bw = calculate_bias(binned_z_pred, binned_z_true)
        scatter_bw = calculate_scatter(binned_z_pred, binned_z_true)
        outlier_bw = calculate_outlier_rate(binned_z_pred, binned_z_true)
        # conventionals
        bias_conv = calculate_bias(binned_z_pred, binned_z_true, conventional=True)
        scatter_conv = calculate_scatter(binned_z_pred, binned_z_true, conventional=True)
        outlier_conv = calculate_outlier_rate(binned_z_pred, binned_z_true, conventional=True)
        # add these all to new row
        metrics_list.append([
            zspec_bin, count, L, bias_bw, bias_conv, 
            scatter_bw, scatter_conv, outlier_bw, outlier_conv])
    # put into a nice df
    metrics_df = pd.DataFrame(metrics_list, columns=[
        'zspec_bin', 'count', 'L', 'bias_bw', 'bias_conv',
        'scatter_bw', 'scatter_conv', 'outlier_bw', 'outlier_conv'])
    return metrics_df


def get_density_metrics(z_photo_vectors, z_spec, binned=False):
    galaxies = z_spec.index
    PIT = calculate_PIT(z_photo_vectors, z_spec)
    CRPS = calculate_CRPS(z_photo_vectors, z_spec)
    metrics_df = pd.DataFrame({'galaxy': galaxies,
                               'PIT': PIT,
                               'CRPS': CRPS})
    return metrics_df


### PLOTTING FUNCTIONS (metrics must already be calculated)


def plot_predictions(z_photo, z_spec):
    sns.set(rc={'figure.figsize':(10,10)})
    sns.histplot(x=z_spec, y=z_photo, cmap='viridis')
    sns.lineplot(x=[0,4], y=[0,4])
    plt.xlabel('True redshift')
    plt.ylabel('Predicted redshift')
    
    
def plot_point_metrics(metrics):
    sns.set(rc={'figure.figsize':(18,8), 'lines.markersize':10})
    bin_lefts = metrics['zspec_bin'].apply(lambda x: x.left)
    sns.lineplot(x=[0,4], y=[0,0], linewidth=2, color='black')
    sns.scatterplot(x=bin_lefts, y=metrics['bias_bw'], marker = '.', edgecolor='none', label='bias')
    sns.scatterplot(x=bin_lefts, y=metrics['bias_conv'], marker = '.', facecolors='none', edgecolor='black')
    sns.scatterplot(x=bin_lefts, y=metrics['scatter_bw'], marker = "s", edgecolor='none', label='scatter')
    sns.scatterplot(x=bin_lefts, y=metrics['scatter_conv'], marker = "s", facecolors='none', edgecolor='black')
    sns.scatterplot(x=bin_lefts, y=metrics['outlier_bw'], marker = "v", edgecolor='none', label='outlier')
    sns.scatterplot(x=bin_lefts, y=metrics['outlier_conv'], marker = "v", facecolors='none', edgecolor='black')
    sns.scatterplot(x=bin_lefts, y=metrics['L'], marker = 'x', label = "loss", linewidth=4)
    plt.xlabel('Redshift')
    plt.ylabel('Statistic value')

    
def plot_density_metrics(metrics):
    sns.set(rc={'figure.figsize':(18,8), 'lines.markersize':10})
    PIT = metrics['PIT']
    CRPS = metrics['CRPS']
    fig, axes = plt.subplots(1,2)
    sns.histplot(PIT, bins = 50, ax=axes[0])
    sns.histplot(CRPS, bins = 50, ax=axes[1])
    plt.yscale('log')

