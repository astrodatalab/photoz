# Data Versions

### HSC_v4
- cleaned_all_specz_flag_forced_forced2_spec_z_matched_online.csv
This file is created from all_specz_flag_forced_forced2_spec_z_matched_online.csv in HSC_v3 with the following filtered applied

1. specz_redshift < 4
2. specz_redshift > 0.01
3. 0 < specz_redshift_err < 1
4. specz_redshift_err < 0.005(1+specz_redshift)
5. the magnitude of each band is between 0 and 50
6. specz_flag_homogeneous == True
