# Results

### Random Forest model
[notebook for random forest model](code/RF_big_data.ipynb)

[results for random forest model](results/RF_results.csv)

- dataset: all_specz_flag_forced_forced2_spec_z_matched_online.csv

- filters: \
specz_redshift < 4\
specz_redshift > 0.01\
0 < specz_redshift_err < 1\
specz_redshift_err < 0.005(1+specz_redshift)\
the magnitude of each band is between 0 and 50

- train test split: 0.2
- metrics:\
bias: 0.0005149503749206011\
scatter: 0.011814389695905432\
biweighht outlier rate: 0.18843685197985582\
conventional outlier rate: 0.022262269181396267\
average loss: 0.039902192290918646


