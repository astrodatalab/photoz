import sys, os
import numpy as np
import pandas as pd
import photoz_utils
import random
import h5py
from astropy.io import fits
from sklearn.preprocessing import MinMaxScaler

def make_hdf5_from_raw_images():
    #Sort the directory and load 
    image_list_g = sorted(os.listdir("/data3/Jonathan/bands/g"))
    image_list_r = sorted(os.listdir("/data3/Jonathan/bands/r"))
    image_list_i = sorted(os.listdir("/data3/Jonathan/bands/i"))
    image_list_z = sorted(os.listdir("/data3/Jonathan/bands/z"))
    image_list_y = sorted(os.listdir("/data3/Jonathan/bands/y"))

    #load metadata
    sheardata = pd.read_csv('/data/HSC/shear_catalog/hsc_shear_with_cuts.csv')
    sheardata.describe()
    
    #get the object IDs from the bands based on the file name {object_id}_{band}image.fits
    objectIDs = []
    for i in range(len(image_list_g)):
        objectIDs.append(int(image_list_g[i][:17]))
        
    sheardata_filter = sheardata[sheardata['object_id'].isin(objectIDs)]
    b = np.argsort(sheardata_filter['object_id'])
    sorted_sheardata = sheardata_filter.iloc[b][:]
    sheardata_filtersorted = sorted_sheardata
    
    #name the file you want to create
    hf = h5py.File('/data3/Jonathan/five_band_image127x127_with__metadata.hdf5', 'a')
    
    for (columnName, columnData) in sheardata_filtersorted.iteritems():
        print(columnName)
        #there was an issue saving hdf5 columns with np Object data type so re-assign the problem columns to str:
        if columnName == 'field':
            a = np.array(sheardata_filtersorted[columnName]).astype(str)
              #b = np.reshape(a,[286401,1])
            b = np.reshape(a,[len(sheardata_filtersorted),1])
            hf.create_dataset(columnName,data=b.astype('S'))
            continue
        hf.create_dataset(columnName, data=sheardata_filtersorted[columnName])
    
    for i in range(len(image_list_g)):    
        five_band_image = []

        image_g = fits.open("/data3/Jonathan/bands/g/"+image_list_g[i])
        image_r = fits.open("/data3/Jonathan/bands/r/"+image_list_r[i])
        image_i = fits.open("/data3/Jonathan/bands/i/"+image_list_i[i])
        image_z = fits.open("/data3/Jonathan/bands/z/"+image_list_z[i])
        image_y = fits.open("/data3/Jonathan/bands/y/"+image_list_y[i])

        image_g_data = image_g[1].data
        image_r_data = image_r[1].data
        image_i_data = image_i[1].data
        image_z_data = image_z[1].data
        image_y_data = image_y[1].data

        pad1 = int((127-len(image_g_data))/2)
        pad2 = 127-len(image_g_data)-pad1
        pad3 = int((127-len(image_g_data[0]))/2)
        pad4 = 127-len(image_g_data[0])-pad3


        im_g = np.pad(image_g_data, ((pad1,pad2), (pad3,pad4)), "constant", constant_values = ((0,0),(0,0)))
        im_r = np.pad(image_r_data, ((pad1,pad2), (pad3,pad4)), "constant", constant_values = ((0,0),(0,0)))
        im_i = np.pad(image_i_data, ((pad1,pad2), (pad3,pad4)), "constant", constant_values = ((0,0),(0,0)))
        im_z = np.pad(image_z_data, ((pad1,pad2), (pad3,pad4)), "constant", constant_values = ((0,0),(0,0)))
        im_y = np.pad(image_y_data, ((pad1,pad2), (pad3,pad4)), "constant", constant_values = ((0,0),(0,0)))

        five_band_image.append(im_g)
        five_band_image.append(im_r)
        five_band_image.append(im_i)
        five_band_image.append(im_z)
        five_band_image.append(im_y)

        five_band_image_reshape = np.reshape(np.array(five_band_image),[1,5,122,122])

        #  sheardata_subset = sheardata_filtersorted.iloc[i]

        #  shear1 = sheardata_subset["ishape_hsm_regauss_e1"]
        #  shear1_reshape = np.reshape(shear1,[1,1])

        if i == 0:
            hf.create_dataset("image", data = five_band_image_reshape, chunks = True, maxshape = (None, 5, 127,127))

        else:
            hf['image'].resize((hf['image'].shape[0]+1), axis=0)
            hf['image'][hf["image"].shape[0]-1,:,:,:] = five_band_image



        image_g.close()
        image_r.close()
        image_i.close()
        image_z.close()
        image_y.close()

    hf.close()
    
    
if __name__ == '__main__':
    make_hdf5_from_raw_images()
