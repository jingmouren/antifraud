def random_forest(X_train, X_test, y_train, y_test):
    '''
    use model RandomForestClassifier to evaluate our data, use X_train, y_train train a model RF, saved in folder model,
    use X_test to test model RF and print the accuracy rate
    Args:
        X_train: features used in model training
        y_train: labels used in model training
        X_test: features used in model testing
        y_test: labels used in model testing
    '''
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    joblib.dump(RF, "model/RF.m")  # save the trained model in folder model as RF.m
    y_pre = RF.predict_proba(X_test)
    y_pred = []
    for item in y_pre:
        if item[0] > item[1]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    print(accuracy_score(y_test, y_pred) * data_ratio)
