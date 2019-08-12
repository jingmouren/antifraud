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
        'gamma': 0.05,  # 树的叶子节点下一个区分的最小损失，越大算法模型越保守
        'max_depth': 12,
        'lambda': 450,  # L2正则项权重
        'subsample': 0.4,  # 采样训练数据，设置为0.5
        'colsample_bytree': 0.7,  # 构建树时的采样比率
        'min_child_weight': 12,  # 节点的最少特征数
        'silent': 1,
        'eta': 0.005,  # 类似学习率
        'seed': 700,
        'nthread': 4,  # cpu线程数
    }
    paras_list = list(paras.items())  # 超参数放到集合plst中;
    offset = int(len(train) * 0.9)  # 训练集中数据, 9/10用于训练，1/10用于验证
    num_rounds = 500  # 迭代次数
    xgb_test = xgb.DMatrix(test)  # 加载数据可以是numpy的二维数组形式，也可以是xgboost的二进制的缓存文件，加载的数据存储在对象DMatrix中
    xgb_train = xgb.DMatrix(train[:offset, :], label=labels[:offset])  # 将训练集的二维数组加入到里面
    xgb_val = xgb.DMatrix(train[offset:, :], label=labels[offset:])  # 将验证集的二维数组形式的数据加入到DMatrix对象中

    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]  # return训练和验证的错误率
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




