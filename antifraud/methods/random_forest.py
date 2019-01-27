# from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def random_forest(train_feature, train_label, test_feature, test_label):
    """
    use model RandomForestClassifier to evaluate our data, use train_feature, train_label train a model rf,
    saved in folder model, use test_feature to test model rf and return the accuracy rate
    Args:
        train_feature: features used in model training
        train_label: labels used in model training
        test_feature: features used in model testing
        test_label: labels used in model testing
    Returns:
        the accuracy score of the fit model
    """

    rf = RandomForestClassifier()
    rf.fit(train_feature, train_label)
    # joblib.dump(rf, "model/rf.m")  # save the trained model in folder model as RF.m
    pre_label = rf.predict_proba(test_feature)
    _pre_label = []
    for item in pre_label:
        if item[0] > item[1]:
            _pre_label.append(0)
        else:
            _pre_label.append(1)
    return accuracy_score(test_label, _pre_label)
