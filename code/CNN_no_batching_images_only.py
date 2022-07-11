import os
from astropy.io import fits
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py
from keras.preprocessing.image import ImageDataGenerator

hf = h5py.File('/home/boscoe/spell/five_band_image127x127_training_full.hdf5', 'r')

# import random
# random_indices = random.sample(range(1, 100000), 5000)

x = hf["image"]
y = hf["specz"]
y_train = np.array(y)


x = np.transpose(x,(0,2,3,1))

max_value = np.max(x)
max_value

x_train = np.true_divide(x,max_value)
hf.close()


hf = h5py.File('/home/boscoe/spell/five_band_image127x127_testing.hdf5', 'r')

# import random
# random_indices = random.sample(range(1, 100000), 5000)

x = hf["image"]
y = hf["specz"]
y_test = np.array(y)


x = np.transpose(x,(0,2,3,1))

max_value = np.max(x)
max_value

x_test = np.true_divide(x,max_value)
hf.close()


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model


input1_ = tf.keras.layers.Input(shape=(127,127,5))
#input2_ = tf.keras.layers.Input(shape=(5,))


#CNN
conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='tanh')(input1_)
pooling1 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv1)
conv2 = tf.keras.layers.Conv2D(32, kernel_size=(2,2),activation='tanh')(pooling1)
pooling2 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv2)
#conv3 = tf.keras.layers.Conv2D(32, kernel_size=(3,3),activation='relu')(pooling2)
conv4 = tf.keras.layers.Conv2D(32, kernel_size=(2,2),activation='relu')(pooling2)
flatten = tf.keras.layers.Flatten()(conv4)
dense1 = tf.keras.layers.Dense(5080, activation="tanh")(flatten)
dense2 = tf.keras.layers.Dense(508, activation="tanh")(dense1)
dense3 = tf.keras.layers.Dense(400,activation = "tanh")(dense2)

output = tf.keras.layers.Dense(1)(dense3)

model = tf.keras.Model(inputs=[input1_],outputs = [output])

model.summary()


model.compile(optimizer='Adam', loss="mse",metrics=[tf.keras.metrics.MeanAbsoluteError()])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
						monitor='loss',
                                                 verbose=1,
                                                save_freq =1000,
						save_best_only=True)


history = model.fit(x = x_train,y = y_train,epochs=300, shuffle = True,verbose=1, callbacks = [cp_callback])


#!mkdir -p saved_model
model.save('saved_model/my_model')
