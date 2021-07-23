# Filters

All the filters are tried out on all_specz_flag_forced_forced2_spec_z_matched_online.csv with 801246 galaxies.

[code/RF_big_data.ipynb](filters.md)


### 1. specz_redshift < 4
- High redshift galaxies are rare and they will not improve the performance of the model. 
- filter out 0.345% galaxies. (2766)


### 2. specz_redshift > 0.01
- This filter is also used by HSC team. Some of the low redshift galaxies are not reliable. And we can not
that intersted in galaxies with extremely small redshift.
- filter out 3.92% galaxies. (31442)


### 3. 0 < specz_redshift_err < 1
- This criteria enforces that the spectroscopic information, which will be used as truth is reliable.
- filtered out 0% galaxies.(0) Since query has already required this.

### 4. specz_redshift_err < 0.005(1+specz_redshift)
- This is another way of enforcing that the speczs we used are reliable.HSC team also used this criteria.
- filtered out 0.995% galaxies. (7972)

### 5. the magnitude of each band is between 0 and 50
- This is trying to get rid of all the nan and infinity in the dataset. Galaxies with entry nan is rare, but they will
ruin the model training process.
- filter out 0.0145% galaxies. (116)
