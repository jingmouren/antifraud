import pandas as pd
import numpy as np
from math import isnan


def data_engineer(data_dir):
    data = pd.read_csv(data_dir)
    data['time_stamp'] = pd.to_datetime(data['time_stamp'])

    time_span = []
    for i in [60, 3600, 86400, 172800, 2628000, 7884000, 15768000, 31536000]:  # 1sec, 1 min, 1 day ...
        time_span.append(pd.Timedelta(seconds=i))

    train = []
    Oct = []
    Nov = []
    Dec = []

    start_time = "2015/1/1 00:00"

    for i in data.iterrows():
        data2 = []
        temp_data = data[data['card_id'] == i[1]['card_id']]
        temp_county_id = i[1]['loc_cty']
        temp_merch_id = i[1]['loc_merch']
        temp_time = i[1]['time_stamp']
        temp_label = i[1]['is_fraud']
        a_grant = i[1]['amt_grant']
        a_purch = i[1]['amt_purch']
        for loc in data['loc_cty'].unique():
            data1 = []
            if (loc in temp_data['loc_cty'].unique()):
                card_tuple = temp_data['loc_cty'] == loc
                single_loc_card_data = temp_data[card_tuple]
                time_list = single_loc_card_data['time_stamp']
                for length in time_span:
                    lowbound = (time_list >= (temp_time - length))
                    upbound = (time_list <= temp_time)
                    correct_data = single_loc_card_data[lowbound & upbound]
                    Avg_grt_amt = correct_data['amt_grant'].mean()
                    Totl_grt_amt = correct_data['amt_grant'].sum()
                    Avg_pur_amt = correct_data['amt_purch'].mean()
                    Totl_pur_amt = correct_data['amt_purch'].sum()
                    Num = correct_data['amt_grant'].count()
                    if (isnan(Avg_grt_amt)):
                        Avg_grt_amt = 0
                    if (isnan(Avg_pur_amt)):
                        Avg_pur_amt = 0
                    data1.append([a_grant, Avg_grt_amt, Totl_grt_amt, a_purch, Avg_pur_amt, Totl_pur_amt, Num])
            else:
                for length in time_span:
                    data1.append([0, 0, 0, 0, 0, 0, 0])
            data2.append(data1)
        if (temp_time > pd.to_datetime(start_time)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=9 * 2628000)):
                train.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=9 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=10 * 2628000)):
                Oct.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=10 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=11 * 2628000)):
                Nov.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=11 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=12 * 2628000)):
                Dec.append([temp_label, np.array(data2)])
    np.save(file='train', arr=train)
    np.save(file='Oct', arr=Oct)
    np.save(file='Nov', arr=Nov)
    np.save(file='Dec', arr=Dec)
    return 0
