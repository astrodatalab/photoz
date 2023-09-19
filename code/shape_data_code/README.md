The README.md file goes over files in the code/shape_data_code directory and how to navigate each file. The folder contains the pipeline used to go from tabular data/image data to testing CNN models on the HSC ML shape dataset. For more information on these files, check out the report in the Google Drive 

#Files in progress
raw_to_hdf5.py
Creates an HDF5 file containing both grizy-band image data and metadata for 100,000 galaxies. The postage stamp size of the images is 127x127. 
The additional 200,000 galaxies need to be added
Ouput file: (Note: name can be changed so it is not confused with the photoz files)
/data3/Jonathan/five_band_image127x127_with__metadata.hdf5

small_subset.py
Takes the large HDF5 file with images and metadata. The ouput of this program is three files pertaining to split of the HDF5 file into a training, a validation, and a test set. The split is roughly 80-10-10 percent of the 100,000 galaxies. The splitting is randomly completed.
The splitting will need to be done witht the full 300,000 galaxy dataset
Output files:
/data3/Jonathan/five_band_image127x127_with__metadata_training.hdf5
/data3/Jonathan/five_band_image127x127_with__metadata_validation.hdf5
/data3/Jonathan/five_band_image127x127_with__metadata_testing.hdf5


hsc_cnn_initial_tests.ipynb
The file contains the final optimal model reached by 9/13/2023 which includes e1/e2 scaled ouputs between 0 and 1, batch size of 32, learning rate of 5e-6, and architecture seen in the file. The file contains two CNNs: VGG19 and GoogLeNet. As of 9/13/2023, GoogLeNet performed the best. The file also plot the predicted vs real e1 values as well as returns residuals, multiplicative bias, and additive bias. 

#Files completed

create_shear_catalog.ipynb
This file took all tabular data downloaded from the HSC Data Acess webpage for each field. It combines all the fields into one large CSV file. In addition, adds a column for total distortion labeled 'e' and a column for what field the galaxy is a part of. All other columns include:
object_id, ira, idec, gcmodel_mag, rcmodel_mag, icmodel_mag, zcmodel_mag, ycmodel_mag, gcmodel_mag_err, rcmodel_mag_err, icmodel_mag_err, zcmodel_mag_err, ycmodel_mag_err, 
ishape_hsm_regauss_e1, ishape_hsm_regaus_e2, ishape_hsm_regauss_derived_shape_weight, ishape_hsm_regauss_derived_shear_bias_m, ishape_hsm_regauss_derived_shear_bias_c1, ishape_hsm_regauss_derived_shear_bias_c2, ishape_hsm_regauss_derived_sigma_e, ishape_hsm_regauss_derived_rms_e, field, e
Ouput file can be found:
/data/HSC/shear_catalog/hsc_shear_no_cuts.csv

shear_catalog_cuts.ipynb
We make cuts on the large shear catalog to match cuts used in the photoz dataset creation. In addition, we test for duplicates and remove them if found.
Output file can be found:
/data/HSC/shear_catalog/hsc_shear_with_cuts.csv

cross_match_hscv6_shear.ipynb
We cross match the galaxies from the HSCv6 dataset and the shear catalog with cuts. We save the cross matched dataset into a file that contains all columns from the shape catalog above plus those in the HSCv6 dataset.
Output file can be found:
/data/HSC/shear_catalog/hsc_v6_with_shape_data.csv

hsc_image_querry.ipynb
The  file generates 3 txt files containing HSC appropriate data querry code. Each individual file contains 100,000 galaxies (an HSC tool limit). We do a grid search around the ra, dec of the galaxy in a 10arcsecond x 10 arcseconds field. We ask that the output FITS files are saved under the format '(object_id)_(grizy)band.FITS'. 

Output file format: (saved to private directory to access on X2Go)
/home/jsoriano/shear_code/image_querry/hsc_shape_batch1{band}.txt
/home/jsoriano/shear_code/image_querry/hsc_shape_batch2{band}.txt
/home/jsoriano/shear_code/image_querry/hsc_shape_batch3{band}.txt
