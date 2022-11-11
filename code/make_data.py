import sys,os
import numpy as np
import pandas as pd
import photoz_utils
import random
import h5py


def make_hsc_v5(inputfile='/mnt/data/HSC/HSC_v3/all_specz_flag_forced_forced2_spec_z_matched_online.csv',
                outputfile='/mnt/data/HSC/HSC_v5/HSC_v5.csv'):
    '''
    Make HSC_v5 from using the same filtering as HSC_v4, but also with duplicate filtering
    
    '''

    photozdata = pd.read_csv(inputfile)

    # rename the object_id column to not include the comment symbol
    photozdata.rename(columns={'# object_id':'object_id'},inplace=True)
    print(photozdata.columns)
    # filter based on data quality criteria
    
    filt = (photozdata['specz_redshift'] < 4)\
    & (photozdata['specz_redshift'] > 0.01)\
    & (photozdata['specz_redshift_err'] > 0) \
    & (photozdata['specz_redshift_err'] < 1)\
    &(photozdata["specz_redshift_err"]<0.005*(1+photozdata["specz_redshift"]))\
    &(photozdata["specz_flag_homogeneous"] == True)\
    &(photozdata['g_cmodel_mag'] >0)\
    &(photozdata['r_cmodel_mag'] >0)\
    &(photozdata['i_cmodel_mag'] >0)\
    &(photozdata['z_cmodel_mag'] >0)\
    &(photozdata['y_cmodel_mag'] >0)\
    &(photozdata['g_cmodel_mag'] < 50)\
    &(photozdata['r_cmodel_mag'] < 50)\
    &(photozdata['i_cmodel_mag'] < 50)\
    &(photozdata['z_cmodel_mag'] < 50)\
    &(photozdata['y_cmodel_mag'] < 50)

    photozdata = photozdata[filt]

    # there are a large number of duplicate spectroscopic matches so we
    # should filter those out

    # first, let's reset the indices after the filtering
    photozdata.reset_index(drop=True,inplace=True)

    duplicate_bool, use_unique_bool = photoz_utils.find_duplicate_galaxies(photozdata)

    print('number of duplicate specz_name galaxies: ',len(np.where(duplicate_bool)[0]))
    print('number of unique specz_name galaxies: ',len(np.where(use_unique_bool)[0]))


    # select only the unique specz_name galaxies
    photozdata = photozdata[use_unique_bool]
    photozdata.reset_index(drop=True,inplace=True)
    
    # now look at duplicate object_ids because there were multiple spectroscopic name matches
    # for this case, we'll remove all the duplicates rather than chosing between the
    # different spectrosopic ones

    duplicate_bool_obj_id, use_unique_bool_obj_id = photoz_utils.find_duplicate_galaxies(photozdata,match_name='object_id')

    print('number of duplicate obj_id galaxies: ',len(np.where(duplicate_bool_obj_id)[0]))
    print('number of unique obj_id galaxies: ',len(np.where(use_unique_bool_obj_id)[0]))
    
    photozdata.drop(np.where(duplicate_bool_obj_id)[0],inplace=True)
    print('number of galaxies to output',len(photozdata))
    print('outputfile: '+outputfile)
    photozdata.to_csv(outputfile,index=None)

def make_hsc_v6(inputfile='/mnt/data/HSC/HSC_v3/all_specz_flag_forced_forced2_spec_z_matched_online.csv',
                outputfile='/mnt/data/HSC/HSC_v6/HSC_v6.csv'):
    '''
    Make HSC_v6 from using the same filtering as HSC_v5, but also with spec redshifts
    
    '''

    photozdata = pd.read_csv(inputfile)

    # rename the object_id column to not include the comment symbol
    photozdata.rename(columns={'# object_id':'object_id'},inplace=True)
    print(photozdata.columns)
    # filter based on data quality criteria
    
    filt = (photozdata['specz_redshift'] < 4)\
    & (photozdata['specz_redshift'] > 0.01)\
    & (photozdata['specz_redshift_err'] > 0) \
    & (photozdata['specz_redshift_err'] < 1)\
    &(photozdata["specz_redshift_err"]<0.005*(1+photozdata["specz_redshift"]))\
    &(photozdata["specz_flag_homogeneous"] == True)\
    &(photozdata['g_cmodel_mag'] >0)\
    &(photozdata['r_cmodel_mag'] >0)\
    &(photozdata['i_cmodel_mag'] >0)\
    &(photozdata['z_cmodel_mag'] >0)\
    &(photozdata['y_cmodel_mag'] >0)\
    &(photozdata['g_cmodel_mag'] < 50)\
    &(photozdata['r_cmodel_mag'] < 50)\
    &(photozdata['i_cmodel_mag'] < 50)\
    &(photozdata['z_cmodel_mag'] < 50)\
    &(photozdata['y_cmodel_mag'] < 50) \
    &(photozdata['specz_mag_i'] < 28) \
    &(photozdata['specz_mag_i'] > 0)

    photozdata = photozdata[filt]

    # there are a large number of duplicate spectroscopic matches so we
    # should filter those out

    # first, let's reset the indices after the filtering
    photozdata.reset_index(drop=True,inplace=True)

    duplicate_bool, use_unique_bool = photoz_utils.find_duplicate_galaxies(photozdata)

    print('number of duplicate specz_name galaxies: ',len(np.where(duplicate_bool)[0]))
    print('number of unique specz_name galaxies: ',len(np.where(use_unique_bool)[0]))


    # select only the unique specz_name galaxies
    photozdata = photozdata[use_unique_bool]
    photozdata.reset_index(drop=True,inplace=True)
    
    # now look at duplicate object_ids because there were multiple spectroscopic name matches
    # for this case, we'll remove all the duplicates rather than chosing between the
    # different spectrosopic ones

    duplicate_bool_obj_id, use_unique_bool_obj_id = photoz_utils.find_duplicate_galaxies(photozdata,match_name='object_id')

    print('number of duplicate obj_id galaxies: ',len(np.where(duplicate_bool_obj_id)[0]))
    print('number of unique obj_id galaxies: ',len(np.where(use_unique_bool_obj_id)[0]))
    
    photozdata.drop(np.where(duplicate_bool_obj_id)[0],inplace=True)
    print('number of galaxies to output',len(photozdata))
    print('outputfile: '+outputfile)
    photozdata.to_csv(outputfile,index=None)
    

def make_hsc_v6_small_subset_hdf(ntrain=10000,ntest=2000,nvalidation=2000,cap=0.1):
    '''
    Create a smaller images dataset for easier training and testing, with maximum z=cap. 


    '''
    sample_sizes = [ntrain,ntest,nvalidation]

    inputfiles = ['five_band_image127x127_with_metadata_corrected_training.hdf5','five_band_image127x127_with_metadata_corrected_testing.hdf5','five_band_image127x127_with_metadata_corrected_validation.hdf5']
    directory = '/mnt/data/HSC/HSC_v6/step2A/127x127/'
    for i in range(len(sample_sizes)):
        current_file = os.path.join(directory,inputfiles[i])
        nsample = sample_sizes[i]
        print(current_file,'nsamples',nsample)
        hf = h5py.File(current_file,'r')
        y_train = np.asarray(hf['specz_redshift'][0:])[..., None]
        inds = np.array([])
        for j in range(len(y_train)):
            if(y_train[j] <= 0.1):
                inds = np.append(inds, j)
            if (len(inds) == nsample):
                break
        inds = inds.astype(int)
        inds = np.sort(inds)
        print(inds)
        part = os.path.splitext(current_file)
        outfile = part[0]+'_0.1_small.hdf5'
        

        f = h5py.File(outfile,'w')
        print('output',outfile)
        for k in hf.keys():

            tmp = hf[k]
            s = list(np.shape(tmp))
            s[0] = nsample
            print(k,nsample,tmp.dtype)            
            dset = f.create_dataset(k,shape=s,dtype=tmp.dtype)
            dset.write_direct(tmp[inds])
        f.close()
        hf.close()
        
        
def make_hsc_v6_subset_hdf(ntrain=10000,ntest=2000,nvalidation=2000,cap=0.1):
    '''
    Create a smaller images dataset for easier training and testing, with maximum z=cap. 


    '''
    sample_sizes = [ntrain,ntest,nvalidation]

    inputfiles = ['five_band_image127x127_with_metadata_corrected_training.hdf5','five_band_image127x127_with_metadata_corrected_testing.hdf5','five_band_image127x127_with_metadata_corrected_validation.hdf5']
    directory = '/mnt/data/HSC/HSC_v6/step2A/127x127/'
    for i in range(len(sample_sizes)):
        current_file = os.path.join(directory,inputfiles[i])
        nsample = sample_sizes[i]
        print(current_file,'nsamples',nsample)
        hf = h5py.File(current_file,'r')
        y_train = np.asarray(hf['specz_redshift'][0:])[..., None]
        inds = np.array([])
        for j in range(len(y_train)):
            if(y_train[j] <= 0.1):
                inds = np.append(inds, j)
        inds = inds.astype(int)
        inds = np.sort(inds)
        print(inds)
        part = os.path.splitext(current_file)
        outfile = part[0]+'_0.1.hdf5'
        

        f = h5py.File(outfile,'w')
        print('output',outfile)
        for k in hf.keys():

            tmp = hf[k]
            s = list(np.shape(tmp))
            s[0] = len(inds)
            print(k,nsample,tmp.dtype)            
            dset = f.create_dataset(k,shape=s,dtype=tmp.dtype)
            dset.write_direct(tmp[inds])
        f.close()
        hf.close()


def make_hsc_v6_small_hdf(ntrain=10000,ntest=2000,nvalidation=2000):
    '''
    Create a smaller images dataset for easier training and testing. 


    '''
    sample_sizes = [ntrain,ntest,nvalidation]

    inputfiles = ['five_band_image127x127_training_with_metadata.hdf5','five_band_image127x127_testing_with_metadata.hdf5','five_band_image127x127_validation_with_metadata.hdf5']
    directory = '/data/HSC/HSC_v6/step2A/127x127/'
    for i in range(len(sample_sizes)):
        current_file = os.path.join(directory,inputfiles[i])
        nsample = sample_sizes[i]
        print(current_file,'nsamples',nsample)
        hf = h5py.File(current_file,'r')
        
        inds = np.sort(random.sample(list(np.arange(len(hf['object_id']))),sample_sizes[i]))

        part = os.path.splitext(current_file)
        outfile = part[0]+'_small.hdf5'
        

        f = h5py.File(outfile,'w')
        print('output',outfile)
        for k in hf.keys():

            tmp = hf[k]
            s = list(np.shape(tmp))
            s[0] = nsample
            print(k,nsample,tmp.dtype)            
            dset = f.create_dataset(k,shape=s,dtype=tmp.dtype)
            dset.write_direct(tmp[inds])
        f.close()
        hf.close()

        
def make_hdf5_from_raw_images():
  #WORKING TO PRODUCE FINAL FULL DATASET:

  #for raw:

  #get number of galaxies in the image directory and sort them

  image_list_g = sorted(os.listdir("../../data/HSC/HSC_v6/step1/g_band"))

  # image_list_r = sorted(os.listdir("../../data/HSC/HSC_v6/step1/r_band"))
  # image_list_i = sorted(os.listdir("../../data/HSC/HSC_v6/step1/i_band"))
  # image_list_z = sorted(os.listdir("../../data/HSC/HSC_v6/step1/z_band"))
  # image_list_y = sorted(os.listdir("../../data/HSC/HSC_v6/step1/y_band"))

  #load metadata
  photozdata = pd.read_csv('/mnt/data/HSC/HSC_v6/HSC_v6.csv')
  photozdata.describe()

  b = np.argsort(photozdata['object_id'])
  sorted_photozdata = photozdata.iloc[b][:]
  photozdata = sorted_photozdata

  #name the file you want to create
  hf= h5py.File('../../data/HSC/HSC_v6/five_band_image127x127_with_metadata_corrected_2.hdf5', 'a')


  for (columnName, columnData) in photozdata.iteritems():
      print(columnName)
      #there was an issue saving hdf5 columns with np Object data type so re-assign the problem columns to str:
      if columnName == 'specz_name' or columnName == 'coord':
              a = np.array(photozdata[columnName]).astype(str)
              #b = np.reshape(a,[286401,1])
              b = np.reshape(a,[286401,1])
              hf.create_dataset(columnName,data=b.astype('S'))

              continue
      hf.create_dataset(columnName,data=photozdata[columnName])


  for i in range(len(image_list_g)):
  #for i in range(10):

      object_id = image_list_g[i][0:17]

      five_band_image = []

      image_g = fits.open("../../data/HSC/HSC_v6/step1/g_band/"+image_list_g[i])
      image_r = fits.open("../../data/HSC/HSC_v6/step1/r_band/"+image_list_r[i])
      image_i = fits.open("../../data/HSC/HSC_v6/step1/i_band/"+image_list_i[i])
      image_z = fits.open("../../data/HSC/HSC_v6/step1/z_band/"+image_list_z[i])
      image_y = fits.open("../../data/HSC/HSC_v6/step1/y_band/"+image_list_y[i])

      image_g_data = image_g[1].data
      image_r_data = image_r[1].data
      image_i_data = image_i[1].data
      image_z_data = image_z[1].data
      image_y_data = image_y[1].data

      pad1 = int((127-len(image_g_data))/2)
      pad2 = 127-len(image_g_data)-pad1
      pad3 = int((127-len(image_g_data[0]))/2)
      pad4 = 127-len(image_g_data[0])-pad3


      im_g = np.pad(image_g_data,((pad1,pad2),(pad3,pad4)),"constant",constant_values = ((0,0),(0,0)))
      im_r = np.pad(image_r_data,((pad1,pad2),(pad3,pad4)),"constant",constant_values = ((0,0),(0,0)))
      im_i = np.pad(image_i_data,((pad1,pad2),(pad3,pad4)),"constant",constant_values = ((0,0),(0,0)))
      im_z = np.pad(image_z_data,((pad1,pad2),(pad3,pad4)),"constant",constant_values = ((0,0),(0,0)))
      im_y = np.pad(image_y_data,((pad1,pad2),(pad3,pad4)),"constant",constant_values = ((0,0),(0,0)))

      five_band_image.append(im_g)
      five_band_image.append(im_r)
      five_band_image.append(im_i)
      five_band_image.append(im_z)
      five_band_image.append(im_y)

      five_band_image_reshape = np.reshape(np.array(five_band_image),[1,5,127,127])

      photozdata_subset = photozdata.iloc[i]

      specz = photozdata_subset["specz_redshift"]
      specz_reshape = np.reshape(specz,[1,1])

      if i == 0:
          hf.create_dataset("image",data = five_band_image_reshape,chunks = True,maxshape = (None,5,127,127))

      else:
          hf['image'].resize((hf['image'].shape[0]+1), axis=0)
          hf['image'][hf["image"].shape[0]-1,:,:,:] = five_band_image



      image_g.close()
      image_r.close()
      image_i.close()
      image_z.close()
      image_y.close()

  hf.close()

def make_hsc_v6_small_hdf_single(ntrain=10000,ntest=2000,nvalidation=2000):
    '''
    Create a smaller images dataset for easier training and testing. 


    '''
    sample_sizes = [ntrain,ntest,nvalidation]

    inputfiles = ['five_band_image127x127_with_metadata_corrected.hdf5']
    directory = '/data/HSC/HSC_v6/step2A/127x127/'
    current_file = os.path.join(directory,inputfiles[0])
    ntrain = sample_sizes[0]
    ntest = sample_sizes[1]
    nvalidation = sample_sizes[2]
    hf = h5py.File('/mnt/data/HSC/HSC_v6/step2A/127x127/five_band_image127x127_with_metadata_corrected.hdf5','r')
        
    inds = random.sample(list(np.arange(len(hf['object_id']))),sample_sizes[0]+sample_sizes[1]+sample_sizes[2])
    inds_train = np.sort(inds[:ntrain])
    inds_test = np.sort(inds[ntrain:ntrain+ntest])
    inds_validation = np.sort(inds[ntrain+ntest:])

    part = os.path.splitext(current_file)
    outfile_train = part[0]+'_training_small.hdf5'
    outfile_test = part[0]+'_testing_small.hdf5'
    outfile_validation = part[0]+'_validation_small.hdf5'
        

    f_train = h5py.File(outfile_train,'w')
    f_test = h5py.File(outfile_test,'w')
    f_validation = h5py.File(outfile_validation,'w')
    for k in hf.keys():

        tmp = hf[k]
        s_train = list(np.shape(tmp))
        s_test = list(np.shape(tmp))
        s_validation = list(np.shape(tmp))
        s_train[0] = ntrain
        s_test[0] = ntest
        s_validation[0] = nvalidation           
        trainset = f_train.create_dataset(k,shape=s_train,dtype=tmp.dtype)
        testset = f_test.create_dataset(k,shape=s_test,dtype=tmp.dtype)
        validationset = f_validation.create_dataset(k,shape=s_validation,dtype=tmp.dtype)
        trainset.write_direct(tmp[inds_train])
        testset.write_direct(tmp[inds_test])
        validationset.write_direct(tmp[inds_validation])
    f_train.close()
    f_test.close()
    f_validation.close()
    hf.close()

    
def make_hsc_v6_large(ntrain=200481,ntest=42960,nvalidation=42960):
    inputfile = 'five_band_image127x127_with_metadata_corrected.hdf5'
    directory = '/data/HSC/HSC_v6/step2A/127x127/'
    current_file = os.path.join(directory, inputfile)
    hf = h5py.File(current_file,'r')
    
    length = len(hf['object_id'])
    inds = random.sample(list(np.arange(length)),ntrain+ntest+nvalidation)
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