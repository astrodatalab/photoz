import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

v1_photoz_path = '/data/HSC/HSC_IMAGES_FIXED/HSC_photozdata_full_header_trimmed.csv'
v2_photoz_path = '/data/HSC_v2/HSC_photozdata_with_spectra.csv'


def import_photoz_data(path='v2'):
    """
    Import the data table of band magnitudes and spectroscopic redshift.

    path: str [optional]
        Path to the dataset you want to import, or alias 'v1' or 'v2' for our
        HSC data versions 1 and 2. Must have the original column names on
        retrieval from the HSC database.
    RETURN: DataFrame
    """
    if (path == 'v1'):
        df = pd.read_csv(v1_photoz_path)
    elif (path == 'v2'):
        df = pd.read_csv(v2_photoz_path)
    else:
        df = pd.read_csv(path)
    return df


def clean_photoz_data(df):
    """
    Clean the data table of band magnitudes and spectroscopic redshift.
    RETURN: DataFrame
    """
    #perform cuts
    cuts = (df['specz_redshift'] < 4) & (df['specz_redshift'] > 0.01) \
            & (df['specz_redshift_err'] > 0) & (df['specz_redshift_err'] < 1)
    df = df[cuts]
    # get magnitudes and spectroscopic redshifts
    df = df[['g_cmodel_mag',
             'r_cmodel_mag',
             'i_cmodel_mag',
             'z_cmodel_mag',
             'y_cmodel_mag',
             'specz_redshift']]
    df.columns = ['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag', 'zspec']
    # remove NAs
    df.replace([-99., -99.9, np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    # summary statistics
    print(df.describe())
    return df


def split_photoz_data(df, test_size=0.2):
    """
    Perform a train-test split of the data.

    df: DataFrame
        The dataframe of photo-z data to split up. Each row represents one
        galaxy, columns are the five-band magnitudes and spectroscopic redshift.
    test_size: float [optional]
    RETURN: tuple
    """

    X = df[['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
    y = df['zspec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def evaluate_point_estimates(z_true, z_pred):
    """
    Evaluate the accuracy between true redshifts and the model's predicted
    point-estimates of each true redshift.

    z_true: (N,) array-like
    z_pred: (N,) array-like
    RETURN: None
    """
    delz = (z_pred-z_true)/(1+z_true)

    r2 = r2_score(z_true, z_pred)
    mse = mean_squared_error(z_true, z_pred)
    bias = sum(delz)/len(delz)
    disp = 1.48*stats.median_abs_deviation(delz)
    print(f'R squared: {r2}\nMean squared error: {mse}\nBias: {bias}\nConventional dispersion: {disp}')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    g = sns.scatterplot(x=z_true, y=z_pred, ax=axes[0,0])
    g.plot([0,4], [0,4], color='red')
    g.set(xlabel='True redshift', ylabel='Predicted redshift')

    g = sns.scatterplot(x=z_true, y=z_pred-z_true, ax=axes[0,1])
    g.plot([0,4], [0,0], color='red')
    g.set(xlabel='True redshift', ylabel='Prediction residual')
    
    g = sns.histplot([z_true, z_pred], bins=100, ax=axes[1,0])
    g.legend(labels=['True redshift', 'Predicted redshift'])
    g.set(xlabel='Redshift')
    
    g = sns.histplot(z_pred-z_true, bins=100, ax=axes[1,1])
    g.set(xlabel='Prediction residual')
    
    plt.show()


def evaluate_interval_estimates(z_true, z_true_err, z_pred, z_pred_err):
    """
    Evaluate the accuracy between true redshifts/redshift errors and the model's
    predicted interval estimates.

    z_true: (N,) array-like
    z_true_err: (N,) array-like
    z_pred: (N,) array-like
    z_pred_err: (N,) array-like
    RETURN: None
    """
    pass


def evaluate_density_estimates(z_true, z_pred_densities):
    """
    Evaluate the accuracy between true redshifts/redshift errors and the model's
    predicted interval estimates.

    z_true: (N,) array-like
    z_pred_densities: (N, M) array-like
        Each row corresponds to a galaxy, containing an (M,) array of guesses
        outputted by the model.
    RETURN: None
    """
    N = len(z_pred_densities)
    M = len(z_pred_densities[0])
    PIT  = []
    CRPS = []
    for z, z_pdf in zip(z_true, z_pred_densities):
        PIT.append(len(np.where(z_pdf<z)[0])*1.0/M)
        for j in range(M):
            z_check = 4.0*j/200
            if z_check < z:
                CRPS.append(((len(np.where(z_pdf<z_check)[0])*1.0/M)**2)*(4.0/200))
            else:
                CRPS.append(((len(np.where(z_pdf<z_check)[0])*1.0/M-1)**2)*(4.0/200))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    g = sns.histplot(PIT, bins = 50, ax=axes[0])
    g.set(xlabel='Probability integral transform', ylabel='Count')
    
    g = sns.histplot(CRPS, bins = 50, ax=axes[1])
    g.set(xlabel='Continuous ranked probability score', ylabel='Count')

    
def evaluate_point_outliers(z_true, z_pred):
    delz = abs(z_pred - z_true)-0.15*(1+z_true)
    is_outlier = (delz > 0)
    num_outliers = sum(is_outlier)*1.0
    outlier_rate = num_outliers/len(z_true)
    L = 1-(1/(1+(delz/0.15)**2))
    print(f'Outlier rate: {outlier_rate}\nLoss function: {L}')
    
    
