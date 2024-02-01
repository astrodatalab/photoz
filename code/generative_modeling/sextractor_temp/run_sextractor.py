import subprocess
import os
import pandas as pd
df = pd.read_csv('/predictions/HSC_v6_NN_neurips_combined_with_5pool_v11/testing_predictions.csv')
obj_ids = df['object_id']
directory = '/data/HSC/HSC_v6/step1/g_band_sextractor/test_set_subset'
for obj_id in obj_ids:
        command = "source-extractor {}_step1.fits -c default2.sex -CATALOG_NAME test_petro_{}.cat -PARAMETERS_NAME default2.param -CHECKIMAGE_NAME test_segmented_{}.fits".format(obj_id, obj_id, obj_id)
        print(command)
        file_path = os.path.join(directory, f'{obj_id}_step1.fits')
        subprocess.run(command + ' ' + file_path, shell=True)
