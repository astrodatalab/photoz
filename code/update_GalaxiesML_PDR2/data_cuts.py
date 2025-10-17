'''
This is a recreation of the cuts implemented in HSC PDR2 for GalaxiesML (Do+24).
'''

import pandas as pd
import numpy as np 

def clean_dataset():
    '''
    Clean initial query from the HSC-PDR3 database. Saves the cleaned dataset 
    with specified cuts.
    '''
    # Load the data
    df = pd.read_csv('/data/HSC/HSC_v10/step0/GalaxiesML_PDR3_step0.csv')
    df.rename(columns={'# object_id': 'object_id'}, inplace=True)
    df.replace([-99., -99.9, np.inf], np.nan, inplace=True)

    print("Initial number of objects:", len(df))

    # Define the cuts
    cut_1 = (df['specz_redshift'] < 4) & (df['specz_redshift'] > 0.01)
    df = df[cut_1]
    print("After cut 1 (specz_redshift < 4 and > 0.01):", len(df))

    cut_2 = (df['specz_redshift_err'] > 0) & (df['specz_redshift_err'] < 1)
    df = df[cut_2]
    print("After cut 2 (0 < specz_redshift_err < 1):", len(df))

    cut_3 = df['specz_redshift_err'] < 0.005 * (1 + df['specz_redshift'])
    df = df[cut_3]
    print("After cut 3 ($\sigma_z$ < 0.005 * (1 + $z_spec$)):", len(df))

    cut_4 = (df['g_cmodel_mag'] > 0) & (df['r_cmodel_mag'] > 0) & \
            (df['i_cmodel_mag'] > 0) & (df['z_cmodel_mag'] > 0) & \
            (df['y_cmodel_mag'] > 0) & (df['g_cmodel_mag'] < 50) & \
            (df['r_cmodel_mag'] < 50) & (df['i_cmodel_mag'] < 50) & \
            (df['z_cmodel_mag'] < 50) & (df['y_cmodel_mag'] < 50)
    df = df[cut_4]
    print("After cut 4 (0 < $grizy$ < 50):", len(df))

    cols = df.columns.tolist()
    cols.remove('specz_mag_i') # ignore i-band magnitude col from specz catalog 
    df.dropna(subset=cols, inplace=True)
    print("After cut 5 (no NaNs in relevant columns):", len(df))

    # Save the cleaned data
    df.to_csv('/data/HSC/HSC_v10/step0/GalaxiesML_PDR3_step1.csv', index=False)
    print("Cleaned data saved to GalaxiesML_PDR3_step1.csv")