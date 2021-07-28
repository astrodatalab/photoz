#goal: visualize photozdaata


import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
#import data set
import numpy as np
mnist = tf.keras.datasets.mnist
import random
from tensorboard.plugins.hparams import api as hp
import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

import seaborn as sns

#import photoz data:
tfd = tfp.distributions
#from google.colab import files


photozdata = pd.read_csv('/data/HSC/HSC_IMAGES_FIXED/HSC_photozdata_full_header_trimmed.csv')
spectro_z = np.asarray(photozdata["specz_redshift"])

col1 = np.asarray(photozdata["g_cmodel_mag"])
col2 = np.asarray(photozdata["r_cmodel_mag"])
col3 = np.asarray(photozdata["i_cmodel_mag"])
col4 =np.asarray(photozdata["z_cmodel_mag"])
col5 = np.asarray(photozdata["y_cmodel_mag"])


#photodata = np.column_stack((col1,col2,col3,col4,col5))

photodata = {'g_mag':col1,
             'r_mag':col2,
             'i_mag':col3,
             'z_mag':col4,
             'y_mag':col5,
             'zspec':spectro_z}

df = pd.DataFrame(photodata)
photozdata = df

spectro_z = pd.DataFrame(spectro_z)

photozdata.describe()

ax = sns.displot(photozdata.zspec)

#Let's also look at the distribution of the brightnesses in the different bands
ax = sns.histplot(photozdata.iloc[:,0:-1], element="step")
ax.set_xlabel('Brightness (mag)')

#Let's examine how correlated the features are to one another
train_df = photozdata.iloc[:,0:-1]
target_df = photozdata.zspec
num_features = len(train_df.columns)
fig,ax = plt.subplots(num_features, num_features, figsize=(5,5))
for ii,i in enumerate(train_df.columns):
    for jj,j in enumerate(train_df.columns):
        if i==j:    # diagonal
            sns.histplot(train_df[i], kde=False, ax=ax[ii][jj])
            ax[ii][jj].axis('off')
        else:       # off diagonal
            sns.scatterplot(x=train_df[i],y=train_df[j], 
                            ax=ax[ii][jj], hue=target_df, palette='BrBG',
                            legend=False)
            ax[ii][jj].axis('off')
fig.tight_layout()
