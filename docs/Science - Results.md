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

