import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats


def import_photoz_data(version=None):
    """
    Import the data table of band magnitudes and spectroscopic redshift.

    path: Path to the dataset or version alias. Must have the original
          column names on retrieval from the HSC database.
    """
    path_dir = '/data/HSC/'
    path_file = 'v' + str(version)
    path = path_dir + path_file
    df = pd.read_csv(path)
    return df


def clean_photoz_data(df, prob_z=False, mizuki_cut=False):
    """
    Clean the data table of band magnitudes and spectroscopic redshift.
    
    df: Non-cleaned dataframe of photo-z data. One row = one galaxy.
    prob_z: True to add an extra err column as output.
    mizuki_cut: True to change error cut to the scaled one used in Mizuki.
    """
    z = df['specz_redshift']
    z_err = df['specz_redshift_err']
    if (mizuki_cut):
        cuts = (z < 4) & (z > 0.01) & (z_err > 0) & (z_err < 0.005*(1+z))
    else:
        cuts = (z < 4) & (z > 0.01) & (z_err > 0) & (z_err < 1)
    df1 = df.loc[cuts]
    df2 = df1.replace([-99., -99.9, np.inf], np.nan)
    df3 = df2.dropna()
    clean_df = df3[['g_cmodel_mag',
             'r_cmodel_mag',
             'i_cmodel_mag',
             'z_cmodel_mag',
             'y_cmodel_mag',
             'specz_redshift']]
    clean_df.columns = ['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag', 'zspec']
    if (prob_z):
        clean_df = clean_df.assign(zspec_err=df3.loc[:,'specz_redshift_err'])
    return clean_df


def split_photoz_data(df, test_size=0.2):
    """
    Perform a train-test split of the data.
    
    df: The clean dataframe of photo-z data. Five inputs, redshift output.
    test_size: Fractional size of the test set.
    """
    
    X = df[['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
    y = df['zspec']
    
    if ('zspec_err' in df.columns):
        e = df['zspec_err']
        X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(X, y, e, test_size=test_size)
        return X_train, X_test, y_train, y_test, e_train, e_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test

