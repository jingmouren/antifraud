from sklearn.ensemble import AdaBoostClassifier
import numpy as np


def adaBM(train_feature_dir, train_label_dir, test_feature_dir, test_label_dir):
    """read data"""
    train_feature = np.load(train_feature_dir)
    train_label = np.load(train_label_dir)
    test_feature = np.load(test_feature_dir)
    test_label = np.load(test_label_dir)
    """ data reshape"""
    train_feature = np.reshape(train_feature, (len(train_feature), 1, 1, 3584))
    test_feature = np.reshape(test_feature, (len(test_feature), 1, 1, 3584))
    new_train_feature = []
    for i in range(len(train_feature)):
        new_train_feature.append(train_feature[i][0][0])
    new_test_feature = []
    for i in range(len(test_feature)):
        new_test_feature.append(test_feature[i][0][0])
    """build the model """
    ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')   # use the default parameters
    """train the model """
    ada.fit(new_train_feature, train_label)
    pre_label = ada.predict_proba(new_test_feature)
    _pre_label = []
    for i in range(len(test_feature)):
        _pre_label.append(pre_label[i][1])
    return np.array(_pre_label), test_label

