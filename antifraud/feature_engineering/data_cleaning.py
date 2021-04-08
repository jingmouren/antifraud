# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:09:02 2019

@author: PENG Feng
@email: im.pengf@foxmail.com
"""

import pandas as pd
import numpy as np
import os
import datetime
import tqdm

from matplotlib import pyplot as plt

# load the original data
data_dir = '../data'
data_name = 'data.csv'
data_types_name = 'data_types.ini'
data_clean_name = 'data_clean.csv'

os.chdir(data_dir)
file_list = os.listdir()

def save_df_as_dict(df, dict_name):
    dict_saved = open(dict_name,'w')
    dict_saved.write(str(dict(df.astype('str'))))
    dict_saved.close()

def read_dict(dict_name):
    dict_read = open(dict_name,'r')
    dict_cont = eval(dict_read.read())
    dict_read.close()
    return dict_cont
    
data_types = read_dict(data_types_name)

if not data_name in file_list:
    print('Pathway Error!')
else:
    os.chdir(data_dir)
    original_data = pd.read_csv(data_name, dtype = data_types)
    original_data_cols = original_data.columns
    
    # dtypes transformation
    original_data[original_data_cols[2]] = pd.to_datetime(original_data[original_data_cols[2]])
    
    memory_usage = np.sum(original_data.memory_usage(deep=True)) / 1024 ** 2
    print('Original data loaded.\n===HEAD===\n{}\n===DTypes===\n{}\nMemory usage: {:.2f} MB'
          .format(original_data.head(5),
                  original_data.dtypes,
                  memory_usage))

# remove cards that haven't had fraud records
fraud_num_sum_by_cards = original_data[['card_id', 'is_fraud']].groupby('card_id').sum().reset_index()
cards_fraud = list(fraud_num_sum_by_cards[fraud_num_sum_by_cards['is_fraud'] > 0]['card_id'])
cards_fraud.sort()
data = original_data[original_data['card_id'].isin(cards_fraud)]

# remove cards that haven't had 1-quarter records
time_stamp_first_trans = data[['card_id', 'time_stamp']]\
    .groupby('card_id').min().reset_index()\
    .rename(columns = {'time_stamp': 'time_stamp_first_trans'})

data = data.merge(time_stamp_first_trans, on = 'card_id', how = 'left')
data['days_from_first_trans'], _ = (data['time_stamp'] - data['time_stamp_first_trans']).astype('str').str.split(" ", n = 1).str
data['days_from_first_trans'] = data['days_from_first_trans'].astype('int')

cards_record_length = data[['card_id', 'days_from_first_trans']].groupby('card_id').max().reset_index()\
    .rename(columns = {'days_from_first_trans': 'record_length'})
cards_more_than_quarter = list(cards_record_length[cards_record_length['record_length'] >= 91]['card_id'])
cards_more_than_quarter.sort()

data = data[data['card_id'].isin(cards_more_than_quarter)]\
    .sort_values(['card_id','time_stamp']).reset_index(drop = True)

# weekday
data['weekday'] = data['time_stamp'].apply(datetime.datetime.weekday)

# hour
data['hour'] = data['time_stamp'].apply(lambda x: datetime.datetime.strftime(x, '%H'))

# feature engineering
# cost a lot of time

# features that is constant
feature_names_trans = ['grant_amt', 'purch_amt', 'rank']
# features that depends on time periods
feature_names_is_most = ['is_most_cty', 'is_most_merch', 'is_most_weekday', 'is_most_hour']
feature_names_per_this = ['per_this_cty', 'per_this_merch', 'per_this_weekday', 'per_this_hour']
feature_names_amt = ['avg_grant_amt', 'avg_purch_amt',
                     'bias_grant_amt', 'bias_purch_amt', 
                     'tol_grant_amt', 'tol_purch_amt']

feature_names = feature_names_trans + feature_names_is_most + feature_names_per_this + feature_names_amt

temporal_labels = ['1_min', '1_hour', '1_day', '1_week', '1_month', '1_quarter', 'all_time']
temporal_timedeltas = [datetime.timedelta(minutes = 1), 
                       datetime.timedelta(hours = 1), 
                       datetime.timedelta(days = 1), 
                       datetime.timedelta(weeks = 1), 
                       datetime.timedelta(days = 30), 
                       datetime.timedelta(days = 91), 
                       datetime.timedelta(days = 9999)]

def is_most(num, col_name, df):
    times = df[['card_id', col_name]].groupby(col_name).count()
    most_times = times.max()
    most_col_name = times[times == most_times].index.values
    return int(num in most_col_name)

def per_this(num, col_name, df):
    num_list = df[col_name]
    if not num in list(num_list):
        return 0.
    else:
        count = len(num_list[num_list == num])
        return count/len(num_list)

def amt_calu(num, col_name, df):
    avg_amt = df[col_name].mean()
    bias_amt = num - avg_amt
    tol_amt = df[col_name].sum()
    return avg_amt, bias_amt, tol_amt

def period_cleaning(period_i, time_stamp, grant_amt, purch_amt, data_clean):
    period = temporal_labels[period_i]
    period_timedelta = temporal_timedeltas[period_i]
    period_threshold = [time_stamp - period_timedelta, time_stamp]

    lines_before_bool_index = (card_record['time_stamp']> period_threshold[0]) & \
                              (card_record['time_stamp']< period_threshold[1])
    if True in list(lines_before_bool_index):
        lines_before = card_record[lines_before_bool_index]
        
        is_most_cty = is_most(card_record.loc[line, 'loc_cty'], 'loc_cty', lines_before)
        is_most_merch = is_most(card_record.loc[line, 'loc_merch'], 'loc_merch', lines_before)
        is_most_weekday = is_most(card_record.loc[line, 'weekday'], 'weekday', lines_before)
        is_most_hour = is_most(card_record.loc[line, 'hour'], 'hour', lines_before)
        per_this_cty = per_this(card_record.loc[line, 'loc_cty'], 'loc_cty', lines_before)
        per_this_merch = per_this(card_record.loc[line, 'loc_merch'], 'loc_merch', lines_before)
        per_this_weekday = per_this(card_record.loc[line, 'weekday'], 'weekday', lines_before)
        per_this_hour = per_this(card_record.loc[line, 'hour'], 'hour', lines_before)
        avg_grant_amt, bias_grant_amt, tol_grant_amt = amt_calu(grant_amt, 'amt_grant', lines_before)
        avg_purch_amt, bias_purch_amt, tol_purch_amt = amt_calu(purch_amt, 'amt_purch', lines_before)
        data_clean.loc[data_clean.shape[0]] = is_fraud, card_id, time_stamp, period, \
            grant_amt, purch_amt, rank, \
            is_most_cty, is_most_merch, is_most_weekday, is_most_hour,\
            per_this_cty, per_this_merch, per_this_weekday, per_this_hour,\
            avg_grant_amt, bias_grant_amt, tol_grant_amt, \
            avg_purch_amt, bias_purch_amt, tol_purch_amt

data_clean = pd.DataFrame(columns = ['is_fraud', 'card_id', 'time_stamp', 'period'] + feature_names)
# progress bar
#import time
pbar = tqdm.tqdm(cards_more_than_quarter)
# every card
for card in pbar:
    pbar.set_description("Processing card: %s" %card)
    #time.sleep(0.01)
    card_record = data[data['card_id'] == card]
    card_record_days = card_record['days_from_first_trans']
    card_record_days_before = card_record_days[card_record_days<30]
    # every transaction
    for line in range(card_record_days_before.index.values[-1], card_record_days.index.values[-1]+1):
        is_fraud, card_id, time_stamp = card_record.loc[line, ['is_fraud', 'card_id', 'time_stamp']]
        grant_amt = card_record.loc[line, 'amt_grant']
        purch_amt = card_record.loc[line, 'amt_purch']
        rank = line
        for period_i in range(len(temporal_labels)):
            period_cleaning(period_i, time_stamp, grant_amt, purch_amt, data_clean)
            
data_clean.to_csv(data_clean_name, index = False)
print('-sys: All work done. Clean data is saved as "{}".'.format(data_clean_name))
