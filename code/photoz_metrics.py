### Evaluating photo-z estimation given true redshifts.
### Metrics match up with those from https://arxiv.org/pdf/2003.01511.pdf

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
        b = stats.median(dz)
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
        eta = stats.mean(outlier_scores > 0.15)
    else:
        b = calculate_bias(z_photo, z_spec)
        s = calculate_scatter(z_photo, z_spec)
        outlier_scores = abs(dz - b)
        eta = stats.mean(outlier_scores > (2*s))
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
