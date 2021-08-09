# Results

### Random Forest model
[notebook for random forest model](../code/RF_big_data.ipynb)

[results for random forest model](../results/RF_results.csv)

- dataset: all_specz_flag_forced_forced2_spec_z_matched_online.csv

- filters: \
specz_redshift < 4\
specz_redshift > 0.01\
0 < specz_redshift_err < 1\
specz_redshift_err < 0.005(1+specz_redshift)\
the magnitude of each band is between 0 and 50\
specz_flag_homogeneous == True

- train test split: 0.2

- metrics:\
biweight bias: 0.000736\
conventional bias: 0.001122\
biweight scatter: 0.026159\
conventional scatter: 0.020999\
biweighht outlier rate: 0.173824\
conventional outlier rate: 0.058884\
average loss: 0.085089

![image](https://user-images.githubusercontent.com/46472635/127441624-27afadf9-99e6-493d-b49c-5e4c928979da.png)
![image](https://user-images.githubusercontent.com/46472635/127441739-b7670937-3b96-4d74-8e93-6250b00b6e7e.png)

### Random Forest model HSC_v6
[notebook for random forest model](../code/RF_big_data.ipynb)

[results for random forest model](../results/RF_results_v6.csv)

- dataset: HSC_v6.csv

- train test split: 0.2

- metrics:\
biweight bias: 0.001666\
conventional bias: 0.002388\
biweight scatter: 0.037309\
conventional scatter: 0.030997\
biweighht outlier rate: 0.179728\
conventional outlier rate: 0.098078\
average loss: 0.127516\
mse: 0.090831

![image](https://user-images.githubusercontent.com/46472635/128378345-5684c871-d426-4190-85e2-c8d161082b82.png)
![image](https://user-images.githubusercontent.com/46472635/128378443-a72490a6-d606-4b7e-9221-bd65b413a23b.png)


### XGBoost model
[notebook for XGBoost model](../code/Photo-Z%20Estimation%20in%20XGBoost.ipynb)


[results for XBGoost model](../results/XGBoost_results.csv)

- dataset: cleaned_all_specz_flag_forced_forced2_spec_z_matched_online.csv

- train test split: 0.2

- metrics:\
biweight bias:0.001656 	\
conventional bias: 0.002814\
biweight scatter: 0.049143\
conventional scatter: 0.041804\
biweighht outlier rate: 0.160973\
conventional outlier rate: 0.109159\
average loss: 0.147154

![image](https://user-images.githubusercontent.com/46472635/127443446-2525d095-3587-4425-98c8-0921b84aa7ad.png)
![image](https://user-images.githubusercontent.com/46472635/127443520-659ecdd5-b526-4ea8-a10c-0f7c0f4e7e50.png)


[results for XBGoost model with data v6](../results/XGBoost_results_v6.csv)
- dataset: HSC_v6.csv

- train test split: 0.2

- metrics:\
biweight bias:0.001776\
conventional bias: 0.002788\
biweight scatter: 0.048867\
conventional scatter: 0.041609\
biweighht outlier rate: 0.162619\
conventional outlier rate: 0.111224\
average loss: 0.14845\
mse: 0.095405


### Neural Network

[notebook for XGBoost model](../code/example_notebook_producing_nans.ipynb )


[results for XBGoost model](../results/NN_regression_results_v6.csv)

- dataset: HSC_v6

- train test split: 0.2

- model: input_ = tf.keras.layers.Input(shape=x_train.shape[1:])\
hidden1 = tf.keras.layers.Dense(400, activation="relu")(input_)\
hidden2 = tf.keras.layers.Dense(400, activation="relu")(hidden1)\
hidden3 = tf.keras.layers.Dense(400, activation="relu")(hidden2)\
hidden4 = tf.keras.layers.Dense(400, activation="relu")(hidden3)\
concat = tf.keras.layers.Concatenate()([input_, hidden4])\
output = tf.keras.layers.Dense(1)(concat)\
model = tf.keras.Model(inputs=[input_], outputs=[output])\
model.compile(optimizer='adam', loss="mse",metrics=[tf.keras.metrics.MeanAbsoluteError()])\


- metrics:\
biweight bias:-0.002323\
conventional bias: -0.002193\
biweight scatter: 0.032684\
conventional scatter: 0.027467\
biweighht outlier rate: 0.150923\
conventional outlier rate: 0.065764\
average loss: 0.098334\
mse: 0.083894

![image](https://user-images.githubusercontent.com/46472635/128579476-6161375d-63d1-4a6f-ab0a-f15c9dad2eba.png)
![image](https://user-images.githubusercontent.com/46472635/128579488-fdd896f3-598b-4591-bc72-6b12e32ea7b8.png)


[results for XBGoost model](../results/NN_regression_custom_loss_results_v6.csv)
This is same as the previous neural network except that the loss function is defined as in the HSC paper. 

- dataset: HSC_v6

- train test split: 0.2

- model: input_ = tf.keras.layers.Input(shape=x_train.shape[1:])\
hidden1 = tf.keras.layers.Dense(400, activation="relu")(input_)\
hidden2 = tf.keras.layers.Dense(400, activation="relu")(hidden1)\
hidden3 = tf.keras.layers.Dense(400, activation="relu")(hidden2)\
hidden4 = tf.keras.layers.Dense(400, activation="relu")(hidden3)\
concat = tf.keras.layers.Concatenate()([input_, hidden4])\
output = tf.keras.layers.Dense(1)(concat)\
model = tf.keras.Model(inputs=[input_], outputs=[output])\
model.compile(optimizer='adam', loss=custom_loss,metrics=[tf.keras.metrics.MeanAbsoluteError()])\


- metrics:\
biweight bias:-0.002836\
conventional bias: -0.002908\
biweight scatter: 0.030743\
conventional scatter: 0.026149\
biweighht outlier rate: 0.150347\
conventional outlier rate: 0.065868\
average loss: 0.096043\
mse: 0.119405
