# Using `photoz_utils.py`

This file has all the functions you should need for building machine learning models for photometric redshift estimation. This is a shorthand document and does not contain the specific information for the inputs. The extended documentation is contained in the `photoz_utils.py` file.

## Data preprocessing

| function | description |
| - | - |
| import_photoz_data(path=None, version=None) | Import the data table of band magnitudes and spectroscopic redshift. Must provide a full path to data or the data version. |
| clean_photoz_data(df, errors=False, filters=None) | Clean the imported dataset. |
| split_photoz_data(df, test_size=0.2) | Split train and test data, return tuple of four or six dataframes depending on whether df had an error column.|

## Point estimate metrics

| function | description |
| - | - |
| delz(z_photo, z_spec) | Errors in prediction. |
| calculate_bias(z_photo, z_spec, conventional=False) | Bias is a measure of center of the distribution of prediction errors. |
| calculate_scatter(z_photo, z_spec, conventional=False) | Scatter is a measure of deviation in the distribution of prediction errors. |
| calculate_outlier_rate(z_photo, z_spec, conventional=False) | Outlier rate is the fraction of prediction errors above a certain level. |
| calculate_loss(z_photo, z_spec) | Loss is accuracy metric defined by HSC, meant to capture the effects of bias, scatter, and outlier all in one. |

## Density estimate metrics
| function | description |
| - | - |
| calculate_PIT(z_photo_vectors, z_spec) | Probability integral transform is the resulting distribution of the CDFs of true redshifts over the whole set of galaxies. A well-calibrated model will have a uniform PIT. This would correspond to small unbiased errors with thin peaks. Slopes in the PIT correspond to biases in the PDFs. |
| calculate_CRPS(z_photo_vectors, z_spec) | Continuous ranked probability score is a measure of error between the predicted galaxy redshift PDF and the actual PDF of galaxy redshifts. | 

## Quick view functions
| function | description |
| - | - |
| get_point_metrics(z_photo, z_spec, binned=False) | Get a dataframe of the point estimate metrics given predictions. |
| get_density_metrics(z_photo_vectors, z_spec) | Get a dataframe of the PIT and CRPS given predictions. |

## Plotting functions
| function | description |
| - | - |
| plot_predictions(z_photo, z_spec) | Plot predicted vs. true redshifts. |
| plot_point_metrics(metrics) | Plot point metrics vs redshift. Must have already generated binned point metrics. |
| plot_density_metrics(metrics) | Plot density metric histograms. Must have already generated density metrics. |
