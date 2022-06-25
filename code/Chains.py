import os
from collections import OrderedDict
import shutil
import numpy as np
from scipy import stats
import pandas as pd

class Chains(object):
    """
    A generic object to store data from MCMC chains. 
    It has convenience methods for savinig the chains to disk and to 
    calculate statistical properties of the chains.

    Based on the MultiNestResult object from StarKit
    """
    

    @classmethod
    def from_array(cls, arr, parameter_names):
        """
        Construct a Chains object from numpy array and list of parameters
        If there is no 'weights' value in the parameter_names, then assume
        the chains are equal weights. 

        Parameters
        ----------
        arr: numpy array (N points,n_parameters)

        Return
        ------
        Chains object
        """
        temp_dict = {}
        for i in np.arange(len(parameter_names)):
            temp_dict[parameter_names[i]] = arr[:,i]
        

        # make an equally weighted chains
        if 'weights' not in parameter_names:
            s = np.shape(arr)
            temp_dict['weights'] = np.ones(s[0])/s[0]

        pd_tab = pd.DataFrame(temp_dict)        

        return cls(pd_tab)

    @classmethod
    def from_dict(cls,d):
        '''
        Construct a Chains object directly from a dictionary. The keys will be 
        the parameter names. If there is no 'weights' value then, assume the
        chains of of equal weights. 

        Parameters
        ----------
        d - dictionary of 1D numpy arrays of the same size

        Return
        ------
        Chains object
        '''
        
        pd_tab = pd.DataFrame(d)
        if 'weights' not in d.keys():
            pd_tab['weights'] = np.ones(len(pd_tab))/float(len(pd_tab))
        
        return cls(pd_tab)


    def __init__(self, posterior_data):
        '''
        Create a Chains object from a pandas data frame. If there are no 'weights' column, then it 
        will create one with the same weights
        '''
        self.posterior_data = posterior_data.copy()

        # figure out what are parameter names. Note all original columns (such as likelihoods) should
        # be accessible through the class variable posterior_data
        self.parameter_names = [col_name for col_name in posterior_data.columns
                                if col_name not in ['x','weights','loglikelihood','lnlike']]
        if 'weights' not in posterior_data.columns:
            s = len(posterior_data)
            self.posterior_data['weights'] = np.ones(s)/float(s)            

    @classmethod
    def from_hdf5(cls, h5_fname, key='chains'):
        """
        Reading a result from its generated HDF5 file

        Parameters
        ----------

        h5_fname: ~str
            HDF5 filename

        key: ~str
            group identifier in the store
        """

        posterior_data = pd.read_hdf(h5_fname, key)

        return cls(posterior_data)


    @property
    def mean(self):
        mean_dict = []
        for param_name in self.parameter_names:
            # sort the parameter in order to create the CDF
            param_x = np.copy(self.posterior_data[param_name])

            weights = np.copy(self.posterior_data['weights'])
            mean = np.average(param_x,weights=weights)
            mean_dict.append((param_name, mean))
            
        return pd.Series(OrderedDict(mean_dict))        


    @property
    def median(self):
        median_dict = []
        for param_name in self.parameter_names:
            # sort the parameter in order to create the CDF
            param_x = np.copy(self.posterior_data[param_name])

            weights = np.copy(self.posterior_data['weights'])
            ind = np.argsort(param_x)
            param_x = np.array(param_x[ind])
            weights = np.array(weights[ind])
            #k = [np.sum(weights[0:i+1]) for i in xrange(len(weights))]

            # make CDF of the weights to determine sigmas later
            k = np.cumsum(weights)
            median = np.interp(0.5,k,param_x)
            median_dict.append((param_name, median))
            
        return pd.Series(OrderedDict(median_dict))


    def __repr__(self):
        m = self.median
        sig = self.calculate_sigmas(1)
        outstr = 'param median -1sigma +1sigma ave_sigma\n'
        for k in sig.keys():
            outstr= outstr+ '%s %f %f %f %f\n' % (k,m[k],sig[k][0],sig[k][1],(sig[k][1]-sig[k][0])/2.0)
        return outstr

    def calculate_sigmas(self, sigma_number):
        sigma_dict = []
        for param_name in self.parameter_names:

            # sort the parameter in order to create the CDF
            param_x = np.copy(self.posterior_data[param_name])

            weights = np.copy(self.posterior_data['weights'])
            ind = np.argsort(param_x)
            param_x = np.array(param_x[ind])
            weights = np.array(weights[ind])
            #k = [np.sum(weights[0:i+1]) for i in xrange(len(weights))]

            # make CDF of the weights to determine sigmas later
            k = np.cumsum(weights)
            sigma_lower = np.interp(stats.norm.cdf(-sigma_number), k, param_x)
            sigma_upper = np.interp(stats.norm.cdf(sigma_number), k, param_x)
            sigma_dict.append((param_name, (sigma_lower, sigma_upper)))
        return OrderedDict(sigma_dict)

    def plot_corner(self,*args,**kwargs):
        '''
        Just a link to plot_triangle for convenience of naming
        '''
        self.plot_triangle(*args,**kwargs)

    def plot_triangle(self, parameters = None, sigma=None,debug=False,show_titles=True,
                      **kwargs):
        '''
        Produce a corner plot of the chains posterior.

        Keywords
        --------
        parameters - a list of paramters to plot. By default, it will plot
                     all fit parameters. This is useful if you run into problems
                     where one of the fit paramters is fixed and corner.py does
                     not work on it
        sigma - limit the plot window to this many sigma from the median. This will 
                help to zoom in on the region of interest for very large number of 
                points. 
        '''
        try:
            from corner import corner
        except ImportError:
            raise ImportError('Plotting requires corner.py')

        if parameters is None:
            plot_params = self.parameter_names
        else:
            plot_params = parameters
        
        if sigma is None:
            range = None
        else:
            sig = self.calculate_sigmas(sigma)
            range = [sig[par] for par in parameters]
            if debug:
                print(range)

        corner(self.posterior_data[plot_params],
               labels=parameters,show_titles=show_titles,
               weights=self.posterior_data['weights'], range=range,**kwargs)
                

    def to_hdf(self, fname_or_buf, key='chains'):
        """
        Writing the result out to HDF5.

        Parameters
        ----------

        fname_or_buf: ~str
            filename or buffer

        key: ~str
            key to save it under default='data'
        """

        self.posterior_data.to_hdf(fname_or_buf, key=key)                                
