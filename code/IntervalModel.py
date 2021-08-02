from RedshiftModel import RedshiftModel


class IntervalEstimator(RedshiftEstimator):
    """
    Class for model that output interval estimates.
    """
    def __init__(self):
        """
        Class-specific data members.
        """
        self.e = None
        self.e_pred = None

        self.e_train = None
        self.e_train_pred = None

        self.e_val = None
        self.e_val_pred = None
        self.e_test_pred = None
    
    ##################
    # DATA FUNCTIONS #
    ##################

    def load_data(self, path=None):
        """
        Load in a dataset from the given path and divide it into inputs and
        outputs. Indexes these arrays by object ID. Shapes of input/output is
        determined by the file type, assuming CSV means 5-band photometry and
        FITS means image data.

        path:       [str] Full path to dataset.
        SETS:       X, y
        RETURNS:    None
        """
    
    def clean_data(self, z_range=(0.01,4), dropna=False, scaled=False):
        """
        Clean the dataset according to specifications.
        
        z_range:    [tuple] Range of redshifts to keep.
        dropna:     [bool]  Whether to drop NAs.
        scaled:     [bool]  Whether to min-max scale over all X values.
        SETS:       X, y
        RETURNS:    None
        """

    def split_data(self, val_size=0.2, test_size=0.2):
        """
        Split data into training, validation, and testing.

        val_size:   [float] Fractional size of the validation set. You may want
                    to set this to 0 if you intend to build your models with
                    cross-validation.
        test_size:  [float] Fractional size of the test set.
        SETS:       X_train, y_train, X_val, y_val, X_test, y_test
        RETURNS:    None
        """

    ###################
    # MODEL FUNCTIONS #
    ###################

    def fit_model(self):
        """
        Fit model using training data. Uses model's fit function.

        SETS:       model
        RETURNS:    None
        """

    ##################
    # OUTPUT RESULTS #
    ##################

    """
    Common variables for this section.

    dataset:  [str] Name of the class dataset to get results for. Options are
                'all', 'train', 'val', and 'test'.
    binned:     [bool] Whether to bin results by spectroscopic redshift bins.
    """

    def print_metrics(self, dataset=None, binned=False):
        """
        Return [DataFrame] of metrics.
        """
        return
        
    def plot_metrics(self, dataset=None):
        """
        Plot metrics as function of true redshift.
        """
        return
    
    def print_predictions(self, dataset=None):
        """
        Return [DataFrame] of predictions. Includes galaxy ID, inputs, true
        redshifts, and predicted redshifts.
        """
        pass

    def plot_predictions(self, dataset=None):
        """
        Plot predicted vs. true redshifts.
        """
        return
