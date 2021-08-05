import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.spatial import cKDTree
#from astropy.coordinates import SkyCoord
#import astropy.units as u

from sklearn.model_selection import train_test_split
from astropy.stats import biweight_location, biweight_midvariance
from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_squared_error



######################
# DATA PREPROCESSING #
######################


def import_photoz_data(path=None, version=None):
    """
    Import the data table of band magnitudes and spectroscopic redshift. Must
    provide a full path to data or the data version. Returns the raw loaded
    dataset as a DataFrame.
    
    path: str
        Full path. Dataset must have the original column names on retrieval from
        the HSC database. Takes precedence over version when both are provided.
    version: int
        Version number. Dataset must have the original column names on retrieval
        from the HSC database. Automatically fills out the pathname for you,
        assuming the data has been named correctly.
    """
    
    if (path is not None):
        pathname = path
    elif (version is not None):
        pathname = '/data/HSC/v' + str(version)
    else:
        sys.exit("Must provide a full path to data or the data version.")
    df = pd.read_csv(pathname)
    return df


def clean_photoz_data(df, errors=False, filters=None, scaled=False):
    """
    Clean the imported dataset. Columns are the band magnitudes, spectroscopic
    redshift, and optionally spectroscopic redshift error. Returns the cleaned
    dataset with specified cuts and NA filtering as a DataFrame.

    df: DataFrame
        Non-cleaned dataframe of photo-z data. Dataset must have the original
        column names on retrieval from the HSC database.
    errors: bool
        Whether to add error column 'specz_err' in the dataframe. Can be used
        later for probabilistic/interval redshift estimation.
    filters: list[int]
        List of ints corresponding to the filters to use. The filters.md file
        contains the list of the applicable filters. If None, no cuts applied.
    scaled: bool
        Whether to scale the magnitude data, min-max across all bands.
    """
    
    # DEFINE CUTS
    cut_1 = (df['specz_redshift'] < 4)
    cut_2 = (df['specz_redshift'] > 0.01)
    cut_3 = (0 < df['specz_redshift_err']) & (df['specz_redshift_err'] < 1)
    cut_4 = df['specz_redshift_err'] < 0.005*(1+df['specz_redshift'])
    cut_5 = (df['g_cmodel_mag'] > 0) & (df['g_cmodel_mag'] < 50)   \
            & (df['r_cmodel_mag'] > 0) & (df['r_cmodel_mag'] < 50) \
            & (df['i_cmodel_mag'] > 0) & (df['i_cmodel_mag'] < 50) \
            & (df['z_cmodel_mag'] > 0) & (df['z_cmodel_mag'] < 50) \
            & (df['y_cmodel_mag'] > 0) & (df['y_cmodel_mag'] < 50)

    # PERFORM CUTS
    if (filters is not None):
        cuts = (cut_1 if 1 in filters else True)    \
                & (cut_2 if 2 in filters else True) \
                & (cut_3 if 3 in filters else True) \
                & (cut_4 if 4 in filters else True) \
                & (cut_5 if 5 in filters else True)
        df = df[cuts]
    
    # NA CUTS
    df = df.replace([-99., -99.9, np.inf], np.nan)
    df = df.dropna()

    # SELECT COLUMNS
    df = df[['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag', 'y_cmodel_mag', 'specz_redshift']]
    df.columns = ['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag', 'z_spec']
    if (errors):
        df = df.assign(z_spec_err=na_df.loc[:,'specz_redshift_err'])
    
    # NORMALIZATION
    if (scaled):
        X = df[['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
        X_norm = (X - X.values.min()) / (X.values.max() - X.values.min())
        df[['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']] = X_norm
    
    return df


def split_photoz_data(df, test_size=0.2):
    """
    Perform a train-test split of the data. Returns tuple of training and test
    datasets.
    
    df: DataFrame
        The clean dataframe of photo-z data. Columns include five band
        magnitudes, spectroscopic redshift, and optionally spectroscopic
        redshift error. 
    test_size: float
        Fractional size of the test set.
    """

    # SELECT INPUTS AND OUTPUTS
    X = df[['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
    z = df['z_spec']

    # SPLIT WITH ERROR
    if ('z_spec_err' in df.columns):
        e = df['z_spec_err']
        X_train, X_test, z_train, z_test, e_train, e_test = train_test_split(X, z, e, test_size=test_size)
        return X_train, X_test, z_train, z_test, e_train, e_test

    # SPLIT WITHOUT ERROR
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size)
    return X_train, X_test, z_train, z_test


def filter_nearest_galaxies(input_csv,output_csv):
    '''
    Examine the data to filter out duplicates based on their positions. TBD

    Inputs:
    ------
    input_csv - input csv table. Should be the csv format for HSC data 
                that our group has been using
    
    Outputs:
    --------
    output_csv - output csv file

    '''
    tab = read_csv(input_csv)



def list_duplicates(seq):
    '''
    Returns a generator with the value and the counts of each
    non-unique element of the input array.

    Inputs:
    --------
    seq - array of values

    Outputs:
    --------
    an interator that returns a list  [item, indices] which are 
    the indices that each duplicate item can be found. 

    '''
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def find_duplicate_galaxies(tab,match_name='specz_name'):
    '''
    Takes in a data frame and determines which values are duplicates. 
    By default will look at the specz_name for duplicates. Will return
    boolean arrays indicating which rows are duplicates and which 
    unique values to use. 

    Inputs:
    -------
    tab - a pandas data frame 

    Keywords:
    ---------
    match_name - the column to use for finding duplicates (default: 'specz_name')

    Output:
    ------
    duplicate_bool, unique_bool - two boolean arrays that indicate 
    which rows are duplicates and which galaxies to use. For galaxies that
    are duplicates, it will assign the first instance of the galaxy to use
    
    '''

    duplicate_gen = list_duplicates(tab[match_name])
    
    duplicate_bool = np.zeros(len(tab),dtype=bool)   # to indicate the galaxy is duplicate
    use_unique_bool = np.ones(len(tab),dtype=bool)   # to indicate the galaxy to use that is unique

    # iterate over the duplicates and just assign one of them to be the true match to the specz
    for name, ind in duplicate_gen:

        duplicate_bool[ind] = True
        # arbitrarily choose the first duplicate to the T one, set all the others to False
        use_unique_bool[ind[1:]] = False
        

    return duplicate_bool, use_unique_bool


##########################
# POINT ESTIMATE METRICS #
##########################


def delz(z_photo, z_spec):
    """
    Returns a vector of residuals/errors in prediction scaled by redshift.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    dz = (z_photo-z_spec) / (1+z_spec)
    return dz


def calculate_bias(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Returns a single value. Bias is a measure of center of the
    distribution of prediction errors.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    conventional: bool
        Whether to use the conventional bias or not. If true, use conventional
        bias, or the median of the errors. If false, use the biweight bias, or
        the biweight location of the errors.
    """
    dz = delz(z_photo, z_spec)
    if (conventional):
        b = np.median(dz)
    else:
        b = biweight_location(dz)
    return b


def calculate_scatter(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Returns a single value. Scatter is a measure of deviation in the
    distribution of prediction errors.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    conventional: bool
        Whether to use the conventional scatter or not. If true, use
        conventional scatter, or the normal MAD of the errors. If false, use the
        biweight bias, or the biweight midvariance of the errors.
    """
    dz = delz(z_photo, z_spec)
    if (conventional):
        s = median_abs_deviation(dz, scale='normal') # normal scale divides MAD by 0.67449
    else:
        s = np.sqrt(biweight_midvariance(dz))
    return s


def calculate_outlier_rate(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Returns a single value. Outlier rate is the fraction of prediction errors above a certain level.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    conventional: bool
        Whether to use the conventional outlier rate or not. If true, use
        conventional outlier rate, or rate of absolute errors above 0.15. If
        false, use the biweight outlier rate, or rate of errors outside two
        deviations of the norm based on the distribution of errors.
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
    HSC METRIC. Returns an array. Loss is accuracy metric defined by HSC, meant
    to capture the effects of bias, scatter, and outlier all in one. This has
    uses for both point and density estimation.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    dz = delz(z_photo, z_spec)
    gamma = 0.15
    denominator = 1.0 + np.square(dz/gamma)
    L = 1 - 1.0 / denominator
    return L


############################
# DENSITY ESTIMATE METRICS #
############################


def calculate_PIT(z_photo_vectors, z_spec):
    """
    HSC METRIC. Returns an array. Probability integral transform is the CDF of
    the CDF of a sampled distribution, used as a measure of the balance between
    the galaxy PDF peak sharpness and the accuracy of the peak. For each galaxy
    PDF, we take the CDF value at the true redshift, that is, the percentile in
    which the true redshift lies in the array of predictions for that galaxy.
    The PIT is the resulting distribution of these galaxy CDF values over the
    whole set of galaxies. A well-calibrated model will have a uniform PIT. This
    would correspond to small unbiased errors with thin peaks. Slopes in the PIT
    correspond to biases in the PDFs.

    z_photo_vectors: array of arrays
        One array of predicted redshifts per galaxy.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    z_spec = np.array(z_spec)
    z_photo_vectors = np.array(z_photo_vectors)
    length = len(z_photo_vectors[0])
    PIT  = np.zeros(len(z_photo_vectors))
    for i in range (len(z_photo_vectors)):          
        PIT[i] += len(np.where(z_photo_vectors[i]<z_spec[i])[0])*1.0/length
    return PIT


def calculate_CRPS(z_photo_vectors, z_spec):
    """
    HSC METRIC. Returns an array. Continuous ranked probability score is a
    measure of error between the predicted galaxy redshift PDF and the actual
    PDF of galaxy redshifts.

    z_photo_vectors: array of arrays
        One array of predicted redshifts per galaxy.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
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


########################
# QUICK VIEW FUNCTIONS #
########################


def get_point_metrics(z_photo, z_spec, binned=False):
    """
    Get a dataframe of the point estimate metrics given predictions.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    binned: bool
        True to calculate metrics by bins. This creates bins of size Δz = 0.2 on
        the spectroscopic redshifts.
    """

    # CREATE BINS
    if (binned):
        bins = pd.cut(z_spec, bins=np.linspace(0, 4, 21))
    else:
        bins = pd.cut(z_spec, bins=np.linspace(0, 4, 2))
    true_grouped = z_spec.groupby(bins)
    pred_grouped = z_photo.groupby(bins)

    # METRICS PER BIN
    metrics_list = []
    for zspec_bin in true_grouped.groups:

        # GET BIN'S PREDICTIONS
        binned_z_true = true_grouped.get_group(zspec_bin)
        binned_z_pred = pred_grouped.get_group(zspec_bin)

        # BASIC STATISTICS
        count = len(binned_z_true)
        L = np.mean(calculate_loss(binned_z_pred, binned_z_true))

        # BIWEIGHT
        bias_bw = calculate_bias(binned_z_pred, binned_z_true)
        scatter_bw = calculate_scatter(binned_z_pred, binned_z_true)
        outlier_bw = calculate_outlier_rate(binned_z_pred, binned_z_true)

        # CONVENTIONAL
        bias_conv = calculate_bias(binned_z_pred, binned_z_true, conventional=True)
        scatter_conv = calculate_scatter(binned_z_pred, binned_z_true, conventional=True)
        outlier_conv = calculate_outlier_rate(binned_z_pred, binned_z_true, conventional=True)
        
        # MSE
        mse = mean_squared_error(binned_z_true,binned_z_pred)

        # ADD TO ROW
        metrics_list.append([
            zspec_bin, count, L, bias_bw, bias_conv, 
            scatter_bw, scatter_conv, outlier_bw, outlier_conv,mse])

    # DATAFRAME CONVERSION
    metrics_df = pd.DataFrame(metrics_list, columns=[
        'zspec_bin', 'count', 'L', 'bias_bw', 'bias_conv',
        'scatter_bw', 'scatter_conv', 'outlier_bw', 'outlier_conv','mse'])
    return metrics_df


def get_density_metrics(z_photo_vectors, z_spec):
    """
    Get a dataframe of the PIT and CRPS given predictions.

    z_photo_vectors: array of arrays
        One array of predicted redshifts per galaxy.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    galaxies = z_spec.index
    PIT = calculate_PIT(z_photo_vectors, z_spec)
    CRPS = calculate_CRPS(z_photo_vectors, z_spec)
    metrics_df = pd.DataFrame({'galaxy': galaxies,
                               'PIT': PIT,
                               'CRPS': CRPS})
    return metrics_df


######################
# PLOTTING FUNCTIONS #
######################


def plot_predictions(z_photo, z_spec):
    """
    Plot predicted vs. true redshifts.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    
    sns.set(rc={'figure.figsize':(10,10)})

    sns.histplot(x=z_spec, y=z_photo, cmap='viridis', cbar=True)
    sns.lineplot(x=[0,4], y=[0,4])
    plt.xlabel('True redshift')
    plt.ylabel('Predicted redshift')
    
    
def plot_point_metrics(metrics):
    """
    Plot binned metrics. Must have already generated point metrics.

    metrics: DataFrame
        Binned point estimate metrics given predictions.
    """

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
    """
    Plot density metrics. Must have already generated density metrics.

    metrics: DataFrame
        Density estimate metrics given predictions.
    """
    
    sns.set(rc={'figure.figsize':(18,8), 'lines.markersize':10})

    PIT = metrics['PIT']
    CRPS = metrics['CRPS']
    fig, axes = plt.subplots(1,2)
    sns.histplot(PIT, bins = 50, ax=axes[0])
    sns.histplot(CRPS, bins = 50, ax=axes[1])
    plt.yscale('log')
