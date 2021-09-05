import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import astropy as ap
from astropy.io import fits
from sklearn.model_selection import train_test_split
import pickle
from astropy.stats import biweight_location, biweight_midvariance
from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_squared_error


class RedshiftModel:
    """
    Base class for all redshift models. Point estimation.
    """
    
    def __init__(self):
        """
        Class data members.
        """
        self.model = None # ML model, compatible with X and y shapes
        self.df = None # original datset
        self.clean_df = None # cleaned dataset
        self.features = None # feature names
        self.outputs = ['specz_redshift'] # output names

        self.X = None # input full set
        self.y = None # output full set
        self.preds = None # predictions on X

        self.X_train = None # input train set
        self.y_train = None # output train set
        self.preds_train = None # predictions on X_train

        self.X_val = None # input validation set
        self.y_val = None # output validation set
        self.preds_val = None # predictions on X_val

        self.X_test = None # input test set
        self.y_test = None # output test set
        self.preds_test = None # predictions on X_test
        

    ##################
    # DATA FUNCTIONS #
    ##################
    
    def load_data(self, path):
        """
        Load in a dataset from the given path and divide it into inputs and
        outputs. Indexes these arrays by object ID. Shapes of input/output is
        determined by the file type, assuming CSV means 5-band photometry and
        FITS means image data.

        path:       [str] Full path to dataset.
        SETS:       X, y
        RETURNS:    None
        """
        df = pd.read_csv(path)
        if ('g_cmodel_mag' in df.columns):
            self.features = ['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag', 'y_cmodel_mag']
        elif ('g_path' in df.columns):
            self.features = ['g_path', 'r_path', 'i_path', 'z_path', 'y_path']
        self.df = df[self.features + self.outputs]
        self.df.index = df['object_id']
        self.X, self.y = self.input_output(self.df)

    
    def clean_data(self, z_range=(0.01,4), dropna=False, scaled=False):
        """
        Clean the dataset according to specifications.
        
        z_range:    [tuple] Range of redshifts to keep.
        dropna:     [bool]  Whether to drop NAs.
        scaled:     [bool]  Whether to min-max scale over all X values.
        SETS:       X, y
        RETURNS:    None
        """
        self.clean_df = self.df
        if (z_range is not None):
            is_inrange = (self.y > z_range[0]) & (self.y < z_range[1])
            self.clean_df = self.clean_df[is_inrange]
        if (dropna):
            is_notna = ~np.any((np.isnan(self.X)) | (self.X == np.inf) | (self.X == -99.) | (self.X == -99.9), axis=1)
            self.clean_df = self.clean_df[is_notna]
        self.X, self.y = self.input_output(self.clean_df)
        if (scaled):
            x_max = np.max(self.X)
            x_min = np.min(self.X)
            self.X = (self.X - x_min) / (x_max - x_min)

    def split_data(self, val_size=0.2, test_size=0.2):
        """
        Split data into training, validation, and testing.

        val_size:   [float] Fractional size of the validation set wrt full set.
        test_size:  [float] Fractional size of the test set wrt full set.
        SETS:       X_train, y_train, X_val, y_val, X_test, y_test
        RETURNS:    None
        """
        val_size = val_size / (1 - test_size)
        X, self.X_test, y, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=val_size)
    

    ###################
    # MODEL FUNCTIONS #
    ###################

    def load(self, path=None, model=None):
        """
        Load in a pickled model or a model instance.

        path:       [str] Full path to saved, pickled model. Takes precedence.
        model:      [obj] Model class. Function creates class instance.
        kwargs:     Arguments you want to pass into the model instantiation.
        SETS:       model
        """
        if(path is not None):
            print("Loading pickled model.")
            self.model = pickle.load(open(path, 'rb'))
        else:
            print("Creating model instance.")
            self.model = model()


    def fit(self, X=None, y=None):
        """
        Fit model using training data. Uses model's own fit function.

        X:          [arr] Training data.
        y:          [arr] Training target.
        SETS:       model
        """
        if ((X is None) | (y is None)):
            print("Fitting on training set.")
            X = self.X_train
            y = self.y_train
        else:
            print("Fitting on passed-in data.")
        self.model.fit(X, y)


    def predict(self, X=None, dataset=None):
        """
        Predict given data.

        X:          [arr] Input data. Takes precedence.
        dataset:    [str] Name of the class dataset to get results for. Options are
                    'all', 'train', 'val', and 'test'. 'all' sets results for all
                    four class datasets.
        RETURNS:    [arr] Predictions from data.
        SETS:       Corresponding preds if dataset name is passed in.
        """
        if ((X is None) & (dataset is not None)):
            print(f"Predicting on {dataset} set.")
            X, y, preds = self.select_dataset(dataset)
            preds = self.model.predict(X)
            self.select_dataset(dataset, set_preds=preds)
            if (dataset == 'all'):
                self.predict('train')
                self.predict('val')
                self.predict('test')
        else:
            print("Predicting on passed-in set.")
            preds = self.model.predict(X)
        return preds


    def save(self, path=None):
        """
        Save model to file.
        """
        try:
            pickle.dump(self.model, open(path, 'wb'))
        except:
            raise ValueError('Must pass in a valid save path to pickle.')
    

    #####################
    # CALCULATE RESULTS #
    #####################
    """
    Common variables for this section.

    y_pred:     [array-like] Photometric or predicted redshifts.
    y_true:     [array-like] Spectroscopic or true redshifts.
    conv:       [bool] Whether to use the conventional outlier rate or not.
    """

    def delz(self, y_pred, y_true):
        """
        Returns [float] delz. A vector of residuals/errors in prediction scaled
        by redshift.
        """
        dz = (y_pred-y_true) / (1+y_true)
        return dz


    def bias(self, y_pred, y_true, conv=False):
        """
        Returns [float] bias. Bias is a measure of center of the distribution of
        prediction errors.
        """
        dz = self.delz(y_pred, y_true)
        b = np.median(dz) if (conv) else biweight_location(dz)
        return b
    

    def scatter(self, y_pred, y_true, conv=False):
        """
        Returns [float] scatter. Scatter is a measure of deviation in the
        distribution of prediction errors.
        """
        dz = self.delz(y_pred, y_true)
        s = median_abs_deviation(dz, scale='normal') if (conv) else np.sqrt(biweight_midvariance(dz))
        return s
    

    def outlier(self, y_pred, y_true, conv=False):
        """
        Returns [float] outlier rate. Outlier rate is the fraction of prediction
        errors above a certain level.
        """
        dz = self.delz(y_pred, y_true)
        if (conv):
            outlier_scores = abs(dz)
            eta = np.mean(outlier_scores > 0.15)
        else:
            b = self.bias(y_pred, y_true)
            s = self.scatter(y_pred, y_true)
            outlier_scores = abs(dz - b)
            eta = np.mean(outlier_scores > (2*s))
        return eta
    

    def loss(self, y_pred, y_true):
        """
        Returns [array] loss by galaxy. Loss is an accuracy metric defined by
        HSC, meant to capture the effects of bias, scatter, and outlier all in
        one. This has uses for both point and density estimation.
        """
        dz = self.delz(y_pred, y_true)
        gamma = 0.15
        denominator = 1.0 + np.square(dz/gamma)
        L = 1 - 1.0 / denominator
        return L
    

    ##################
    # OUTPUT RESULTS #
    ##################
    """
    Common variables for this section.

    y_pred:     [arr] Predicted or photometric redshifts. Takes precedence if
                passed with y_true.
    y_true:     [arr] True or spectroscopic redshifts. Takes precedence if
                passed with y_pred.
    dataset:    [str] Name of the class dataset to get results for. Options are
                'all', 'train', 'val', and 'test'. If used, must have already
                used predict() on the given dataset.
    binned:     [bool] Whether to bin results by spectroscopic redshift bins.
    """

    def print_metrics(self, y_pred=None, y_true=None, dataset=None, binned=False):
        """
        Return [DataFrame] of metrics.
        """
        if ((y_pred is None) | (y_true is None)): # use dataset name
            X, y_true, y_pred = self.select_dataset(dataset)
            if (y_pred is None):
                sys.exit("Prediction has not been done on this set.")
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)

        bin_num = 21 if (binned) else 2
        bins = pd.cut(y_true, bins=np.linspace(0, 4, bin_num))
        true_grouped = y_true.groupby(bins)
        pred_grouped = y_pred.groupby(bins)

        metrics_list = []
        for zspec_bin in true_grouped.groups:
            # GET BIN'S PREDICTIONS
            binned_z_true = true_grouped.get_group(zspec_bin)
            binned_z_pred = pred_grouped.get_group(zspec_bin)
            # BASIC STATISTICS
            count = len(binned_z_true)
            L = np.mean(self.loss(binned_z_pred, binned_z_true))
            # BIWEIGHT
            bias_bw = self.bias(binned_z_pred, binned_z_true)
            scatter_bw = self.scatter(binned_z_pred, binned_z_true)
            outlier_bw = self.outlier(binned_z_pred, binned_z_true)
            # CONVENTIONAL
            bias_conv = self.bias(binned_z_pred, binned_z_true, conv=True)
            scatter_conv = self.scatter(binned_z_pred, binned_z_true, conv=True)
            outlier_conv = self.outlier(binned_z_pred, binned_z_true, conv=True)
            # MSE
            mse = mean_squared_error(binned_z_true,binned_z_pred)
            # ADD TO ROW
            metrics_list.append([
                zspec_bin, count, L, bias_bw, bias_conv, 
                scatter_bw, scatter_conv, outlier_bw, outlier_conv,mse])
        # DATAFRAME CONVERSION
        metrics_df = pd.DataFrame(metrics_list, columns=[
            'zspec_bin', 'count', 'L', 'bias_bw', 'bias_conv',
            'scatter_bw', 'scatter_conv', 'outlier_bw', 'outlier_conv','mse'])
        return metrics_df
        
        
    def plot_metrics(self, y_pred=None, y_true=None, dataset=None):
        """
        Plot metrics as function of true redshift.
        """
        metrics = self.print_metrics(y_pred, y_true, dataset, binned=True)
        sns.set(rc={'figure.figsize':(18,8), 'lines.markersize':10})
        bin_lefts = metrics['zspec_bin'].apply(lambda x: x.left)
        sns.lineplot(x=[0,4], y=[0,0], linewidth=2, color='black')
        sns.scatterplot(x=bin_lefts, y=metrics['bias_bw'], marker = '.', edgecolor='none', label='bias')
        sns.scatterplot(x=bin_lefts, y=metrics['bias_conv'], marker = '.', facecolors='none', edgecolor='black')
        sns.scatterplot(x=bin_lefts, y=metrics['scatter_bw'], marker = "s", edgecolor='none', label='scatter')
        sns.scatterplot(x=bin_lefts, y=metrics['scatter_conv'], marker = "s", facecolors='none', edgecolor='black')
        sns.scatterplot(x=bin_lefts, y=metrics['outlier_bw'], marker = "v", edgecolor='none', label='outlier')
        sns.scatterplot(x=bin_lefts, y=metrics['outlier_conv'], marker = "v", facecolors='none', edgecolor='black')
        sns.scatterplot(x=bin_lefts, y=metrics['L'], marker = 'x', label = "loss", linewidth=4)
        plt.xlabel('Redshift')
        plt.ylabel('Statistic value')

    
    def print_predictions(self, y_pred=None, y_true=None, dataset=None):
        """
        Return [DataFrame] of predictions. Includes true redshifts and predicted
        redshifts. If dataset name is passed in, includes galaxy id and
        features.
        """
        if ((y_pred is None) | (y_true is None)): # use dataset name
            X, y_true, y_pred = self.select_dataset(dataset)
            if (y_pred is None):
                sys.exit("Prediction has not been done on this set.")
        output_dict = {'object_id': self.id, 'X': self.X, 'y_true': self.y, 'y_pred': self.preds}
        output_df = pd.DataFrame(output_dict)
        return output_df



    def plot_predictions(self, y_pred=None, y_true=None, dataset=None):
        """
        Plot predicted vs. true redshifts.
        """
        if ((y_pred is None) | (y_true is None)): # use dataset name
            X, y_true, y_pred = self.select_dataset(dataset)
            if (y_pred is None):
                sys.exit("Prediction has not been done on this set.")
        sns.set(rc={'figure.figsize':(10,10)})
        sns.histplot(x=y_true, y=y_pred, cmap='viridis', cbar=True)
        sns.lineplot(x=[0,4], y=[0,4])
        plt.xlabel('True redshift')
        plt.ylabel('Predicted redshift')

    ####################
    # HELPER FUNCTIONS #
    ####################

    def select_dataset(self, dataset, set_preds=None):
        """
        Helper function for choosing data by name.
        """
        if (dataset == 'all'):
            self.preds = set_preds if (set_preds is not None) else self.preds
            return self.X, self.y, self.preds
        if (dataset == 'train'):
            self.preds_train = set_preds if (set_preds is not None) else self.preds_train
            return self.X_train, self.y_train, self.preds_train
        if (dataset == 'val'):
            self.preds_val = set_preds if (set_preds is not None) else self.preds_val
            return self.X_val, self.y_val, self.preds_val
        if (dataset == 'test'):
            self.preds_test = set_preds if (set_preds is not None) else self.preds_test
            return self.X_test, self.y_test, self.preds_test

    def input_output(self, df):
        """
        Helper function for df to input/output numpy arrays.
        """
        X = df[self.features].to_numpy()
        y = df[self.outputs].to_numpy().flatten()
        return X, y