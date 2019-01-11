from antifraud.__main__ import logger


def logistic(X_train, X_test, y_train, y_test):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    '''
    use model LogisticRegression to evaluate our data, use X_train, y_train train a model LR, saved in folder model,
    use X_test to test model LR and print the accuracy rate
    Args:
        X_train: features used in model training
        y_train: labels used in model training
        X_test: features used in model testing
        y_test: labels used in model testing
    '''
    sc = StandardScaler()  # use StandardScaler to normalize our data for the model
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    LR = LogisticRegression(C=1000.0, random_state=0)  # use the default parameters
    LR.fit(X_train_std, y_train)
    joblib.dump(LR, "model/LR.m")  # save the trained model in folder model as LR.m
    y_pre = LR.predict_proba(X_test_std)
    y_pred = []
    for item in y_pre:
        if item[0] > item[1]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    logger.info(accuracy_score(y_test,y_pred) * data_ratio)
