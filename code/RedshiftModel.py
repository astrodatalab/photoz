class RedshiftEstimator:
    """
    Base class for all redshift models. Point estimation.
    """
    
    def __init__(self):
        """
        Class data members.
        """
        self.model = None # ML model, compatible with X and y shapes

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

    def set_model(self, model):
        """
        Set our internal model.

        model:      [ML Model] Any kind of machine learning model. Architecture
                    should be complete.
        SETS:       model
        RETURNS:    None
        """

    def fit_model(self):
        """
        Fit model using training data. Uses model's fit function.

        SETS:       model
        RETURNS:    None
        """
    
    #####################
    # CALCULATE RESULTS #
    #####################
    
    """
    Common variables for this section.

    z_photo:    [array-like] Photometric or predicted redshifts.
    z_spec:     [array-like] Spectroscopic or true redshifts.
    conv:       [bool] Whether to use the conventional outlier rate or not.
    """

    def delz(self, z_photo, z_spec):
        """
        Returns [float] delz. A vector of residuals/errors in prediction scaled
        by redshift.
        """
        return
    
    def bias(self, z_photo, z_spec, conv=False):
        """
        Returns [float] bias. Bias is a measure of center of the distribution of
        prediction errors.
        """
        return
    
    def scatter(self, z_photo, z_spec, conv=False):
        """
        Returns [float] scatter. Scatter is a measure of deviation in the
        distribution of prediction errors.
        """
        return
    
    def outlier(self, z_photo, z_spec, conv=False):
        """
        Returns [float] outlier rate. Outlier rate is the fraction of prediction
        errors above a certain level.
        """
        return
    
    def loss(self, z_photo, z_spec):
        """
        Returns [array] loss by galaxy. Loss is an accuracy metric defined by
        HSC, meant to capture the effects of bias, scatter, and outlier all in
        one. This has uses for both point and density estimation.
        """
        return
    
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

    
