from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import numpy as np
def transf(feature):
    ff = json.loads(feature.replace("'",'"'))
    ff = pd.DataFrame(ff)
    return ff.values

def re_transf(feature_list):
    return_list = []
    for item in feature_list:
        return_list.append(json.dumps(item))
    return return_list

def fraud_split(X,y, Cost):
    fraud = []
    nonfr = []
    cost_of_fraud = []
    for i in range(len(y)):
        label = y[i]
        feature = X[i]
        if label == 0:
            nonfr.append(feature)
        else:
            fraud.append(feature)
            cost_of_fraud.append(Cost[i])
    return nonfr,fraud,cost_of_fraud

def generate_new_samples(fraud_set, cost_of_fraud, labels, rate=70):
    augmented_fraud = fraud_set
    label_set = {}
    label_list = []
    length = len(fraud_set)
    cost_array = np.array(cost_of_fraud)
    sample_coef = length * rate / cost_array.sum()

    def generate_random_sample(item1, item2):
        item1 = pd.DataFrame(item1)
        item2 = pd.DataFrame(item2)
        alpha = np.random.random()
        sample = alpha*item1 + (1-alpha)*item2
        return sample.values
    for i in range(length):
        label = str(labels[i])
        cost_and_feature = {'cost':cost_of_fraud[i], 'feature':fraud_set[i]}
        if label not in label_list:
            label_list.append(label)
            label_set[label] = [cost_and_feature]
        else:
            label_set[label].append(cost_and_feature)
    for index_ in label_set:
        item = label_set[index_]
        label_amount = len(item)
        for i in range(label_amount):
            feature = item[i]['feature']
            sample_num = int(sample_coef * item[i]['cost'])
            for j in range(sample_num):
                _index = np.random.randint(label_amount)
                new_x = generate_random_sample(feature, item[_index]['feature'])
                augmented_fraud.append(new_x)
    return augmented_fraud

def save_data_file(feature, label, out_name='data.csv'):
    output = pd.DataFrame({'label':label, 'feature_matrix':[json.dumps(item.tolist()) for item in feature]})
    output.to_csv(out_name)
    print("Successfully generate the data file: {}".format(out_name))

def combind_train_test_split(comb_s,comb_t,test_size=0.25):
    a = len(comb_s)
    if a != len(comb_t):
        print("Input error! Two lists to be combinded aren't length equal!")
    tmp_X = [[comb_s[i],comb_t[i]] for i in range(a)]
    train_X, test_X = train_test_split(tmp_X, test_size=test_size)
    train_s =[]
    test_s = []
    train_t = []
    test_t = []
    for item in train_X:
        train_s.append(item[0])
        train_t.append(item[1])
    for item in test_X:
        test_s.append(item[0])
        test_t.append(item[1])
    return train_s, test_s, train_t, test_t

def get_x_y(filepath = './transaction.csv'):
    a = pd.read_csv(filepath)
    y_train = a['label'].tolist()
    x_train = a['feature_matrix'].tolist()
    x_train = [transf(item) for item in x_train]#time consuming
    return x_train, y_train

def kgn_main(x_train = None, y_train=None, Cost=None):
    a = x_train
    if a == None:
        print("Start load input data files...")
        x_train, y_train = get_x_y(filepath= './transaction.csv')
    k = 12
    if Cost == None:
        c = pd.read_excel('./cost.0.25.xlsx')
        Cost = c[0].tolist()
    print("Start preprocessing...")
    non_fraud_set, fraud_set, cost_of_fraud = fraud_split(x_train, y_train, Cost)

    train_non_fraud, test_non_fraud = train_test_split(non_fraud_set, test_size=0.25)

    train_fraud, test_fraud, train_cost, test_cost = combind_train_test_split(fraud_set, cost_of_fraud)

    scaled_test_fraud_set = scale(np.array(test_fraud).reshape((1,len(test_fraud),9*9))[0])
    print("Start kmeans process...")
    test_kmeans = KMeans(n_clusters = k, random_state = 0).fit(scaled_test_fraud_set)
    test_labels = test_kmeans.labels_
    print("Start generate new samples...")

    augmented_test_fraud = generate_new_samples(test_fraud, test_cost, test_labels)#time consuming

    feature_test = []
    len_ft = 0
    label_test = []
    for item in augmented_test_fraud:
        len_ft += 1
        location = np.random.randint(0,len_ft)
        feature_test.insert(location,item)
        label_test.insert(location,1)
    for item in test_non_fraud:
        len_ft += 1
        location = np.random.randint(0,len_ft)
        feature_test.insert(location,item)
        label_test.insert(location,0)

    scaled_fraud_set = scale(np.array(train_fraud).reshape((1,len(train_fraud),9*9))[0])
    print("Start kmeans process...")
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(scaled_fraud_set)
    labels = kmeans.labels_
    print("Start generate new samples...")

    augmented_fraud = generate_new_samples(train_fraud, train_cost, labels)#time consuming

    feature_train = []
    len_ft = 0
    label_train = []
    for item in augmented_fraud:
        len_ft += 1
        location = np.random.randint(0,len_ft)
        feature_train.insert(location,item)
        label_train.insert(location,1)
    for item in train_non_fraud:
        len_ft += 1
        location = np.random.randint(0,len_ft)
        feature_train.insert(location,item)
        label_train.insert(location,0)
    train_filename = 'train_set.csv'
    test_filename = 'test_set.csv'
    save_data_file(feature_train, label_train, train_filename)
    save_data_file(feature_test, label_test, test_filename)
    print("Finished generating data set!")
    return feature_train, label_train, feature_test, label_test


if __name__ == '__main__':
    kgn_main()
