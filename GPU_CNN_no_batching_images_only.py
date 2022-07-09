
import os
from astropy.io import fits
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
      # Restrict TensorFlow to only allocate 10GB of memory on the first GPU
     try:
         tf.config.experimental.set_virtual_device_configuration( gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=300)])
         tf.config.experimental.set_virtual_device_configuration( gpus[1], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=300)])
         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
         #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                             
     except RuntimeError as e:
     # Virtual devices must be set before GPUs have been initialized
         print(e)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py
#from tensorflow import keras
#from keras.preprocessing.image import ImageDataGenerator


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
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


input1_ = tf.keras.layers.Input(shape=(127,127,5))
#input2_ = tf.keras.layers.Input(shape=(5,))


#CNN
conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='tanh')(input1_)
pooling1 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv1)
conv4 = tf.keras.layers.Conv2D(32, kernel_size=(2,2),activation='relu')(pooling1)
flatten = tf.keras.layers.Flatten()(conv4)
dense1 = tf.keras.layers.Dense(5080, activation="tanh")(flatten)
dense2 = tf.keras.layers.Dense(508, activation="tanh")(dense1)
output = tf.keras.layers.Dense(1)(dense2)
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
                                                 save_freq = 1000,
                                                 save_best_only=True)


history = model.fit(x = x_train,y = y_train,batch_size= 1, epochs=1, shuffle = True,verbose=1, callbacks = [cp_callback])


#!mkdir -p saved_model
model.save('saved_model/my_model')
