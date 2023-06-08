This README.md file will go over navigating this directory and discuss pertinent files. The main purpose of this directory is conducting analysis on galaxy image properties.
Any files in this directory that I do not mention that were created before this README file can be assumed deprecated.

# File still in progress. Updates to come soon.

make_shape_info_hdf5.ipynb
This notebook adds morphological parameters to the pre-existing dataset. It includes already-present metadata columns. Each band has its own set of morphological parameters. I recommend the use of principal component analysis in the future to reduce the number of features.
All of the columns included in the HDF5 file are: 'coord', 'dec', 'g_central_image_pop_10px_rad', 'g_central_image_pop_15px_rad', 'g_central_image_pop_5px_rad', 'g_cmodel_mag', 'g_cmodel_magsigma', 'g_ellipticity', 'g_half_light_radius', 'g_isophotal_area', 'g_major_axis', 'g_minor_axis', 'g_peak_surface_brightness', 'g_petro_rad', 'g_pos_angle', 'g_sersic_index', 'i_central_image_pop_10px_rad', 'i_central_image_pop_15px_rad', 'i_central_image_pop_5px_rad', 'i_cmodel_mag', 'i_cmodel_magsigma', 'i_ellipticity', 'i_half_light_radius', 'i_isophotal_area', 'i_major_axis', 'i_minor_axis', 'i_peak_surface_brightness', 'i_petro_rad', 'i_pos_angle', 'i_sersic_index', 'image', 'object_id', 'r_central_image_pop_10px_rad', 'r_central_image_pop_15px_rad', 'r_central_image_pop_5px_rad', 'r_cmodel_mag', 'r_cmodel_magsigma', 'r_ellipticity', 'r_half_light_radius', 'r_isophotal_area', 'r_major_axis', 'r_minor_axis', 'r_peak_surface_brightness', 'r_petro_rad', 'r_pos_angle', 'r_sersic_index', 'ra', 'skymap_id', 'specz_dec', 'specz_flag_homogeneous', 'specz_mag_i', 'specz_name', 'specz_ra', 'specz_redshift', 'specz_redshift_err', 'x_coord', 'x_coord_x', 'x_coord_y', 'y_central_image_pop_10px_rad', 'y_central_image_pop_15px_rad', 'y_central_image_pop_5px_rad', 'y_cmodel_mag', 'y_cmodel_magsigma', 'y_coord', 'y_coord_x', 'y_coord_y', 'y_ellipticity', 'y_half_light_radius', 'y_isophotal_area', 'y_major_axis', 'y_minor_axis', 'y_peak_surface_brightness', 'y_petro_rad', 'y_pos_angle', 'y_sersic_index', 'z_central_image_pop_10px_rad', 'z_central_image_pop_15px_rad', 'z_central_image_pop_5px_rad', 'z_cmodel_mag', 'z_cmodel_magsigma', 'z_ellipticity', 'z_half_light_radius', 'z_isophotal_area', 'z_major_axis', 'z_minor_axis', 'z_peak_surface_brightness', 'z_petro_rad', 'z_pos_angle', 'z_sersic_index'
I will make a README file in the future to discuss the morphological shape parameters.
Full output file found in /data/HSC/HSC_v6/step2/127x127/five_band_image127x127_with_metadata_with_morphology.hdf5

make_shape_info_hdf5_subsets.ipynb
Adds morphological parameters to the pre-existing training, validation, and testing HDF5 files found in /data/HSC/HSC_v6/step2A/127x127/.
Full output files found in:
/data/HSC/HSC_v6/step2A/127x127/5x127x127_training_with_morphology.hdf5
/data/HSC/HSC_v6/step2A/127x127/5x127x127_validation_with_morphology.hdf5
/data/HSC/HSC_v6/step2A/127x127/5x127x127_testing_with_morphology.hdf5

make_shape_info_hdf5_NORMALIZEDsubsets.ipynb
Acts like the previous notebook but the morphological parameters and magnitudes are normalized using sklearn.preprocessing.MinMaxScaler.
Output files found in:
/data/HSC/HSC_v6/step2A/127x127/5x127x127_training_with_morphology_normalized.hdf5
/data/HSC/HSC_v6/step2A/127x127/5x127x127_validation_with_morphology_normalized.hdf5
/data/HSC/HSC_v6/step2A/127x127/5x127x127_testing_with_morphology_normalized.hdf5

run_sextractor.ipynb
This notebook has a template script that is used in the command line to run Source Extractor.

make_sextractor_subsets.ipynb
Makes training, validation, and testing subdirectories for each band in /data/HSC/HSC_v6/step1/{band}_sextractor.
Source Extractor is currently ran in each individual subdirectory. (Total of 15, 5 bands and 3 subsets)

shape_feature_extraction.ipynb
Reads catalog and segmented image files created by Source Extractor and saves their morphological information in CSV files. I used 5 CSV files, one for each band.
Re-run the notebook for individual bands as to not try to read everything at once.
Output files found in:
/data/HSC/HSC_v6/step1/{band}_band_sextractor/shape_parameters.csv
