### Evaluating photo-z estimation given true redshifts.
### Metrics match up with those from https://arxiv.org/pdf/2003.01511.pdf

import pandas as pd
import numpy as np
import astropy.stats as astrostats
import scipy.stats as stats

def delz(z_photo, z_spec):
    """
    Vector of residuals.
    """
    dz = (z_photo-z_spec) / (1+z_spec)
    return dz


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


def calculate_PIT(z_photo_vectors, z_spec):
    PIT  = np.zeros(len(z_photo_vectors))
    for i in range (len(z_photo_vectors)):          
        PIT[i] = len(np.where(z_photo_vectors[i]<z_spec[i])[0])*1.0/len(z_photo_vectors[0])
    return PIT


def calculate_CRPS(z_photo_vectors, z_spec):
    length = len(z_photo_vectors[0])
    crps = np.zeros(len(z_photo_vectors))
    for i in range (len(z_photo_vectors)):
        for j in range (200):
            z = 4.0*j/200
            if z < z_spec[i]:
                crps[i] += ((len(np.where(z_photo_vectors[i]<z)[0])*1.0/length)**2)*(4.0/200)
            else:
                crps[i] += ((len(np.where(z_photo_vectors[i]<z)[0])*1.0/length-1)**2)*(4.0/200)
    return crps


def print_point_metrics(z_photo, z_spec, binned=False):
    if (binned):
        bins = pd.cut(z_spec, bins=np.linspace(0, 4, 21))
        true_grouped = z_spec.groupby(bins)
        pred_grouped = z_photo.groupby(bins)
        binned_stats_rows = []
        
        for z_bin in true_grouped.groups:
            binned_z_true = true_grouped.get_group(z_bin)
            binned_z_pred = pred_grouped.get_group(z_bin)
            count = len(binned_z_true)
            b = calculate_bias(binned_z_pred, binned_z_true)
            s = calculate_scatter(binned_z_pred, binned_z_true)
            eta_conv = calculate_outlier_rate(binned_z_pred, binned_z_true, conventional=True)
            eta_bw = calculate_outlier_rate(binned_z_pred, binned_z_true)
            loss = np.mean(calculate_loss(binned_z_pred, binned_z_true))
            binned_stats_rows.append([z_bin, count, b, s, eta_conv, eta_bw, loss])
        
        binned_stats = pd.DataFrame(binned_stats_rows, columns=['bin', 'count', 'bias', 'scatter', 'outlier_conv', 'outlier_bw', 'mean_loss'])
        binned_stats.index = binned_stats['bin']
        del binned_stats['bin']
        print(binned_stats)
        
    else:
            b = calculate_bias(z_photo, z_spec)
            s = calculate_scatter(z_photo, z_spec)
            eta_conv = calculate_outlier_rate(z_photo, z_spec, conventional=True)
            eta_bw = calculate_outlier_rate(z_photo, z_spec)
            L = calculate_loss(z_photo, z_spec)
            print(f"Bias: {b}")
            print(f"Scatter: {s}")
            print(f"Conventional outlier rate: {eta_conv}")
            print(f"Biweight outlier rate: {eta_bw}")
            print(f"Losses: \n{L}")