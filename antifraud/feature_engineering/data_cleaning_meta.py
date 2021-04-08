# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:09:02 2019

@author: PENG Feng
@email: im.pengf@foxmail.com
"""

import pandas as pd
import numpy as np
import os

# load the original data
data_dir = '../data'
data_name = 'original_data.csv'
data_types_name = 'data_types.ini'

data_saved_name = 'data.csv'

file_list = os.listdir(data_dir)
if not data_name in file_list:
    print('Pathway Error!')
else:
    os.chdir(data_dir)
    original_data = pd.read_csv(data_name, index_col=0).reset_index(drop = True)
    '''
    # dtype transformation
    original_data['time'] = pd.to_datetime(original_data['time'], unit = 'ns')
    '''
    memory_usage = np.sum(original_data.memory_usage(deep=True)) / 1024 ** 2
    print('-sys: Original data loaded.\n===HEAD===\n{}\n===DTypes===\n{}\nMemory usage: {:.2f} MB'
          .format(original_data.head(5),
                  original_data.dtypes,
                  memory_usage))

feature_names = ['is_fraud', 'card_id', 'time_stamp', 'loc_cty', 'loc_merch', 'amt_grant', 'amt_purch']
#temporal_labels = ['1 min', '1 hour', '1 day', '1 week', '1 month', '1 quarter']
original_data.columns
original_data.head(5)

data = pd.DataFrame([])
data[feature_names[0]] = original_data['label']

_, data[feature_names[1]] = original_data['card_id'].str.split("_").str
data[feature_names[1]] = data[feature_names[1]].astype('int')

data[feature_names[2]] = pd.to_datetime(original_data['time'])

_, data[feature_names[3]] = original_data['county_id'].str.split("_").str
data[feature_names[3]] = data[feature_names[3]].astype('int')

_, data[feature_names[4]] = original_data['merch_id'].str.split("_").str
data[feature_names[4]] = data[feature_names[4]].astype('int')

data[feature_names[5]] = original_data['amt_grant']

data[feature_names[6]] = original_data['amt_purch']

memory_usage = np.sum(data.memory_usage(deep=True)) / 1024 ** 2
print('-sys: Data Cleaning Finished.\n===HEAD===\n{}\n===DTypes===\n{}\nMemory usage: {:.2f} MB'
      .format(data.head(5),
              data.dtypes,
              memory_usage))

with open(data_types_name,'w') as data_types_saved:
    data_types = str(dict(data.dtypes.astype('str')))
    data_types = data_types.replace('datetime64[ns]', 'str')
    data_types_saved.write(data_types)

data.to_csv(data_saved_name, index = False)
print('-sys: All work done. Data is saved as "{}", Datatypes is saved as "{}".'
      .format(data_saved_name,
              data_types_name))