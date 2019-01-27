from antifraud.methods.deep_forest.GCForest import gcForest
from sklearn.metrics import accuracy_score


def deep_forest(train_feature, train_label, test_feature, test_label):
    """
    use model deep-forest to evaluate our data, use train_feature, train_label train a model deep_forest,
    saved in folder model, use test_feature to test model deep_forest and return the accuracy rate
    Args:
        train_feature: features used in model training
        train_label: labels used in model training
        test_feature: features used in model testing
        test_label: labels used in model testing
    Returns:
        the accuracy score of the fit model
    """
    clf = gcForest(shape_1X=(1, 3), window=[2])
    clf.fit(train_feature, train_label)
    pre_label = clf.predict(test_feature)
    return accuracy_score(test_label, pre_label)



