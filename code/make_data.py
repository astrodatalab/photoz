import sys
import numpy as np
import pandas as pd
import photoz_utils



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
    
