import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

v1_photoz_path = '/data/HSC/HSC_IMAGES_FIXED/HSC_photozdata_full_header_trimmed.csv'
v2_photoz_path = '/data/HSC/HSC_v2/HSC_photozdata_with_spectra.csv'
v3_photoz_path = '/data/HSC/HSC_v3/matched_photozdata_with_spectrozdata_full_unfiltered_readable.csv'

def import_photoz_data(path='v3'):
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
    elif (path == 'v3'):
        df = pd.read_csv(v3_photoz_path)
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