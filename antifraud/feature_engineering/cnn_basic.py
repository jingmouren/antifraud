import json
import pandas as pd
import time
mode1 = "%Y-%m-%d %H:%M:%S"
mode2 = "%Y/%m/%d %H:%M"

def transf_time(timestamp, mode=mode1):
    timearray = time.strptime(str(timestamp), mode)
    return int(time.mktime(timearray))

def get_json(data_file = 'filter_data.csv'):
    output_card = 'card.json'
    data_all = pd.read_csv(data_file, sep=",")
    #data_all = data_all[:10]
    #json_example = {
    #    'c_0000':[
    #        {'label':0 ,'time':123, 'amt_grant':10.0, 'amt_purch':10.0, 'county_id':'c_11'},
    #        {'label':0 ,'time':225, 'amt_grant':11.0, 'amt_purch':11.0, 'county_id':'c_11'}
    #        ],
    #    'c_0001':[
    #        {'label':0 ,'time':123, 'amt_grant':10.0, 'amt_purch':10.0, 'county_id':'c_11'},
    #        {'label':0 ,'time':225, 'amt_grant':11.0, 'amt_purch':11.0, 'county_id':'c_11'}
    #        ]
    #}

    card_j = {}
    merch_j = {}
    for index,item in data_all.iterrows():
        #print('All the type is:',type(item['label']),type(item['card_id']),type(item['time']),type(item['amt_grant']),type(item['county_id']))
        #print('The item is:',item)
        card_id = item['card_id']
        merch_id = item['merch_id']
        tmplabel = item['label']
        tmptime = item['time']
        tmpgrant = item['amt_grant']
        tmppurch = item['amt_purch']
        tmpct = item['county_id']
        obj = {'label':tmplabel, 'time':tmptime, 'merch_id':merch_id,'amt_grant':tmpgrant, 'amt_purch':tmppurch, 'county_id':tmpct}
        try:
            tmp = card_j[card_id]
            tmp.append(obj)
            card_j[card_id] = tmp
        except:
            card_j[card_id] = [obj]
    with open(output_card,'w') as out_c:
        out_c.write(json.dumps(card_j))
    return output_card

def get_featrue_map_bak(json_file='card.json'):
    with open(json_file,'r') as json_f:
        contents = json_f.read()
    contents = json.loads(contents)
    print('Successfully load json data! \nStart generate feature matrix.')
    def get_feature(list):
        sorted_list = sorted(list,key=lambda transaction: transaction['time'])
        feature_list=[]
        full_index = []
        list_length = len(sorted_list)
        country_set = {}
        merch_set = {}
        for i in range(list_length):
            item = sorted_list[i]
            time_i = item['time']
            try:
                tmp = country_set[item['county_id']]
                tmp += 1
                country_set[item['county_id']] = tmp
            except:
                country_set[item['county_id']] = 1
            try:
                tmp = merch_set[item['merch_id']]
                tmp += 1
                merch_set[item['merch_id']] = tmp
            except:
                merch_set[item['merch_id']] = 1
            index_list = [[],[],[],[],[],[],[],[],[]]
            for j in range(i+1):
                time_j = sorted_list[j]['time']
                time_bias = transf_time(time_i)-transf_time(time_j)
                if time_bias < 63072000:
                    index_list[8].append(j)
                if time_bias < 31536000:
                    index_list[7].append(j)
                if time_bias < 15768000:
                    index_list[6].append(j)
                if time_bias < 7884000:
                    index_list[5].append(j)
                if time_bias < 2628000:
                    index_list[4].append(j)
                if time_bias < 604800:
                    index_list[3].append(j)
                if time_bias < 172800:
                    index_list[2].append(j)
                if time_bias < 86400:
                    index_list[1].append(j)
                if time_bias < 3600:
                    index_list[0].append(j)
            full_index.append(index_list)
        max_country = max(country_set,key=country_set.get)
        max_merch = max(merch_set,key=merch_set.get)
        for i in range(list_length):
            tmp_obj = {'label':None, 'feature_matrix':None}
            tmp_obj['label'] = sorted_list[i]['label']
            matrix_all = []
            for indexs in full_index[i]:
                feature_item = {'Avg_grt_amt':0, 'Totl_grt_amt':0, 'Bias_grt_amt':0, 'Avg_pur_amt':0, 'Totl_pur_amt':0, 'Bias_pur_amt':0, 'Num':0, 'Most_cty':0, 'Most_merch':0}
                total_grant = 0
                total_purch = 0
                purch_num = 0
                for _index in indexs:
                    purch_num += 1
                    total_grant += sorted_list[_index]['amt_grant']
                    total_purch += sorted_list[_index]['amt_purch']
                feature_item['Avg_grt_amt'] = float(total_grant)/purch_num
                feature_item['Totl_grt_amt'] = total_grant
                feature_item['Bias_grt_amt'] = sorted_list[i]['amt_grant'] - feature_item['Avg_grt_amt']
                feature_item['Avg_pur_amt'] = float(total_purch)/purch_num
                feature_item['Totl_pur_amt'] = total_purch
                feature_item['Bias_pur_amt'] = sorted_list[i]['amt_purch'] - feature_item['Avg_pur_amt']
                feature_item['Num'] = purch_num
                if sorted_list[i]['county_id'] == max_country:
                    feature_item['Most_cty'] = 1
                if sorted_list[i]['merch_id'] == max_merch:
                    feature_item['Most_merch'] = 1
                matrix_all.append(feature_item)
            tmp_obj['feature_matrix'] = matrix_all
            feature_list.append(tmp_obj)
        return feature_list
    final_data = []
    for item in contents:
        print("Start get the feature matrix of card: ",item)
        tmp_data = get_feature(contents[item])
        final_data.extend(tmp_data)
    print("Finished generate all feature matrix, now start generate output file.")
    return final_data

def get_featrue_map(json_file='card.json'):
    with open(json_file,'r') as json_f:
        contents = json_f.read()
    contents = json.loads(contents)
    print('Successfully load json data! \nStart generate feature matrix.')
    def get_feature(list):
        sorted_list = sorted(list,key=lambda transaction: transaction['time'])
        feature_list=[]
        full_index = []
        list_length = len(sorted_list)
        ##country_set = {}
        ##merch_set = {}
        for i in range(list_length):
            item = sorted_list[i]
            time_i = item['time']
            #try:
            #    tmp = country_set[item['county_id']]
            #    tmp += 1
            #    country_set[item['county_id']] = tmp
            #except:
            #    country_set[item['county_id']] = 1
            #try:
            #    tmp = merch_set[item['merch_id']]
            #    tmp += 1
            #    merch_set[item['merch_id']] = tmp
            #except:
            #    merch_set[item['merch_id']] = 1
            index_list = [[],[],[],[],[],[],[],[],[]]
            for j in range(i+1):
                time_j = sorted_list[j]['time']
                time_bias = transf_time(time_i)-transf_time(time_j)
                if time_bias < 63072000:
                    index_list[8].append(j)
                if time_bias < 31536000:
                    index_list[7].append(j)
                if time_bias < 15768000:
                    index_list[6].append(j)
                if time_bias < 7884000:
                    index_list[5].append(j)
                if time_bias < 2628000:
                    index_list[4].append(j)
                if time_bias < 604800:
                    index_list[3].append(j)
                if time_bias < 172800:
                    index_list[2].append(j)
                if time_bias < 86400:
                    index_list[1].append(j)
                if time_bias < 3600:
                    index_list[0].append(j)
            full_index.append(index_list)
        #max_country = max(country_set,key=country_set.get)
        #max_merch = max(merch_set,key=merch_set.get)
        for i in range(list_length):
            tmp_obj = {'label':None, 'feature_matrix':None}
            tmp_obj['label'] = sorted_list[i]['label']
            matrix_all = []
            for indexs in full_index[i]:
                feature_item = {'Avg_grt_amt':0, 'Totl_grt_amt':0, 'Bias_grt_amt':0, 'Avg_pur_amt':0, 'Totl_pur_amt':0, 'Bias_pur_amt':0, 'Num':0, 'Most_cty':0, 'Most_merch':0}
                total_grant = 0
                total_purch = 0
                purch_num = 0
                country_set = {}
                merch_set = {}
                for _index in indexs:
                    purch_num += 1
                    total_grant += sorted_list[_index]['amt_grant']
                    total_purch += sorted_list[_index]['amt_purch']
                    try:
                        country_set[sorted_list[_index]['county_id']] += 1
                    except:
                        country_set[sorted_list[_index]['county_id']] = 1
                    try:
                        merch_set[sorted_list[_index]['merch_id']] += 1
                    except:
                        merch_set[sorted_list[_index]['merch_id']] = 1
                max_country = max(country_set,key=country_set.get)
                max_merch = max(merch_set,key=merch_set.get)
                feature_item['Avg_grt_amt'] = float(total_grant)/purch_num
                feature_item['Totl_grt_amt'] = total_grant
                feature_item['Bias_grt_amt'] = sorted_list[i]['amt_grant'] - feature_item['Avg_grt_amt']
                feature_item['Avg_pur_amt'] = float(total_purch)/purch_num
                feature_item['Totl_pur_amt'] = total_purch
                feature_item['Bias_pur_amt'] = sorted_list[i]['amt_purch'] - feature_item['Avg_pur_amt']
                feature_item['Num'] = purch_num
                if sorted_list[i]['county_id'] == max_country:
                    feature_item['Most_cty'] = 1
                if sorted_list[i]['merch_id'] == max_merch:
                    feature_item['Most_merch'] = 1
                matrix_all.append(feature_item)
            tmp_obj['feature_matrix'] = matrix_all
            feature_list.append(tmp_obj)
        return feature_list
    final_data = []
    for item in contents:
        print("Start get the feature matrix of card: ",item)
        tmp_data = get_feature(contents[item])
        final_data.extend(tmp_data)
    print("Finished generate all feature matrix, now start generate output file.")
    return final_data


def feature_engineering(mode=0, filename = 'filter_data.csv'):
    if mode == 0:
        card_f = get_json(filename)
        final_data = get_featrue_map(card_f)
        output = pd.DataFrame(final_data)
        output.to_csv('transaction.csv')
        #output.to_json('transaction.json')
    else:
        card_f = get_json('filter_data_test.csv')
        final_data = get_featrue_map(card_f)
        output = pd.DataFrame(final_data)
        output.to_csv('transaction_test.csv')
        #output.to_json('transaction_test.json')


if __name__ == '__main__':
    feature_engineering(mode=0, filename = 'filter_data.csv')
    #to be continued
