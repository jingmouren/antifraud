from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


def xgb_model(train_feature_dir, train_label_dir, test_feature_dir, test_label_dir):
    train_feature = np.load(train_feature_dir)
    train_label = np.load(train_label_dir)
    test_feature = np.load(test_feature_dir)
    test_label = np.load(test_label_dir)

    train_feature = np.reshape(train_feature, (len(train_feature), 1, 1, 3584))
    test_feature = np.reshape(test_feature, (len(test_feature), 1, 1, 3584))
    new_train_feature = []
    for i in range(len(train_feature)):
        new_train_feature.append(train_feature[i][0][0])
    new_test_feature = []
    for i in range(len(test_feature)):
        new_test_feature.append(test_feature[i][0][0])

    Gbdt = GradientBoostingClassifier(random_state=10)  # use the default parameters

    Gbdt.fit(new_train_feature, train_label)

    pre_label = Gbdt.predict_proba(new_test_feature)
    _pre_label = []
    for i in range(len(test_feature)):
        _pre_label.append(pre_label[i][1])
    return np.array(_pre_label), test_label





