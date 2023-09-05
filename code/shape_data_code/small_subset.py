import sys, os
import numpy as np
import pandas as pd
import photoz_utils
import random
import h5py
from astropy.io import fits

def make_subsets(ntrain=80000,ntest=10000,nvalidation=10000):
    inputfile = 'five_band_image127x127_with_metadata.hdf5'
    directory = '/data3/Jonathan/'
    current_file = os.path.join(directory, inputfile)
    hf = h5py.File(current_file,'r')
    
    length = len(hf['object_id'])
    inds = random.sample(list(np.arange(length)), ntrain + ntest + nvalidation)
    inds_train = np.sort(inds[:ntrain])
    inds_test = np.sort(inds[ntrain:ntrain+ntest])
    inds_validation = np.sort(inds[ntrain+ntest:])

    part = os.path.splitext(current_file)
    subsizes = [ntrain, ntest, nvalidation]
    file_ends = ['_training', '_testing', '_validation']
    ind_list = [inds_train, inds_test, inds_validation]
    
    for subsize, file_end, ind in zip(subsizes, file_ends, ind_list):
        f = h5py.File(part[0] + file_end + part[1], 'w')
        for k in hf.keys():
            tmp = hf[k]
            subshape = list(np.shape(tmp))
            subshape[0] = subsize
            dataset = f.create_dataset(k,shape=subshape,dtype=tmp.dtype)
            for i, index in enumerate(ind):
                dataset[i] = tmp[index]
            tmp = None
        f.close()
    hf.close()
        
    
if __name__ == '__main__':
    make_subsets()
