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

### HSC_v5

Same filters as HSC_v4 except with extra conditions on duplicates:

1. For galaxies where there are multiple matches to one spectroscopic redshift, select the first matching HSC galaxy.
2. For the case where a single HSC galaxy matches multiple spectroscopic values, remove that galaxy.

NOTE: also, the first line no longer contains the comment # so the column is now properly named 'object_id'

number of duplicate specz_name galaxies:  565185
number of unique specz_name galaxies:  291231
number of duplicate obj_id galaxies:  3457
number of unique obj_id galaxies:  289483
number of galaxies to output 287774
outputfile: /mnt/data/HSC/HSC_v5/HSC_v5.csv

### HSC_v6

Same filters as HSC_v5 with conditions on specz_i_mag:

1. specz_i_mag > 0 # this is to get rid of negative values
2. specz_i_mag < 28 # this matches the faintest i band magnitude from photometric measurements, which is about 27.9
