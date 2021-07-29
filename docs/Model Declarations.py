class RedshiftModel:
    """
    Base class for all redshift models.
    """
    
    def __init__(self):
        """
        Class data members.
        """
        self.df = None # full imported dataframe, input+output
        self.clean_df = None # full cleaned dataframe, input+output
        
        self.X = None # input
        self.y = None # output
        self.y_pred = None # predictions on X
        
        self.X_train = None # input train set
        self.y_train = None # output train set
        self.y_train_pred = None # predictions on X_train
        
        self.X_val = None # input validation set
        self.y_val = None # output validation set
        self.y_val_pred = None # predictions on X_val

        self.X_test = None # input test set
        self.y_test = None # output test set
        self.y_test_pred = None # predictions on X_test
        
        self.model = None # the machine learning model, compatible with X and y shapes
        return
    
    ##################
    # DATA FUNCTIONS #
    ##################
    
    @abstractmethod
    def import_data(self, path=None):
        pass
    
    @abstractmethod
    def clean_data(self):
        pass
    
    @abstractmethod
    def split_data(self):
        pass
    
    ###################
    # MODEL FUNCTIONS #
    ###################
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self, dataset='train'):
        """
        Save predictions according to specified dataset.
        dataset (str): 'train', 'validation', or 'test'
        """
        pass
    
    #####################
    # CALCULATE RESULTS #
    #####################
    
    @classmethod
    def delz(self):
        pass
    
    @classmethod
    def bias(self, conv=False):
        pass
    
    @classmethod
    def scatter(self, conv=False):
        pass
    
    @classmethod
    def outlier(self, conv=False):
        pass
    
    @classmethod
    def loss(self):
        pass
    
    ##################
    # OUTPUT RESULTS #
    ##################
    
    @classmethod
    def print_metrics(self):
        pass
    
    @classmethod
    def print_predictions(self):
        pass
    
    @classmethod
    def plot_metrics(self):
        pass

    @classmethod
    def plot_predictions(self):
        pass
    
    ################
    # SAVE RESULTS #
    ################
    
    @classmethod
    def save_predictions(self):
        pass
    
    @classmethod
    def save_metrics(self):
        pass