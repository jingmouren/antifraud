# -*- coding=utf-8 -*-
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score


def xgb_model(train_file, test_file):
    dataset = pd.read_table(train_file, sep=' ', header=None)
    train = dataset.iloc[:, 1:].values
    labels = dataset.iloc[:, :1].values
    tests = pd.read_table(test_file, sep=' ', header=None)
    test = tests.iloc[:, 1:].values
    test_labels = tests.iloc[:, :1].values
    paras = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.05,
        'max_depth': 12,
        'lambda': 450,
        'subsample': 0.4,
        'colsample_bytree': 0.7,
        'min_child_weight': 12,
        'silent': 1,
        'eta': 0.005,
        'seed': 700,
        'nthread': 4,
    }
    paras_list = list(paras.items())
    offset = int(len(train) * 0.9)
    num_rounds = 500
    xgb_test = xgb.DMatrix(test)
    xgb_train = xgb.DMatrix(train[:offset, :], label=labels[:offset])
    xgb_val = xgb.DMatrix(train[offset:, :], label=labels[offset:])

    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(paras_list, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    pre_result = model.predict(xgb_test, ntree_limit=model.best_iteration)
    pre_label = []
    for item in pre_result:
        if item > 0.5:
            pre_label.append(1.0)
        else:
            pre_label.append(0.0)
    _label = []
    for item in test_labels:
        _label.append(item[0])
    return accuracy_score(_label, pre_label)




