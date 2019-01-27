# from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def logistic(train_feature, train_label, test_feature, test_label):
    """
    use model LogisticRegression to evaluate our data, use train_feature, train_label train a model lr,
    saved in folder model, use test_feature to test model lr and return the accuracy rate
    Args:
        train_feature: features used in model training
        train_label: labels used in model training
        test_feature: features used in model testing
        test_label: labels used in model testing
    Returns:
        the accuracy score of the fit model
    """
    sc = StandardScaler()  # use StandardScaler to normalize our data for the model
    sc.fit(train_feature)
    train_feature_std = sc.transform(train_feature)
    test_feature_std = sc.transform(test_feature)
    lr = LogisticRegression(C=1000.0, random_state=0)  # use the default parameters
    lr.fit(train_feature_std, train_label)
    # joblib.dump(lr, "model/lr.m")  # save the trained model in folder model as LR.m
    pre_label = lr.predict_proba(test_feature_std)
    _pre_label = []
    for item in pre_label:
        if item[0] > item[1]:
            _pre_label.append(0)
        else:
            _pre_label.append(1)
    return accuracy_score(test_label, _pre_label)
