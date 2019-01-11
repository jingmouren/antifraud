from antifraud.__main__ import logger
from sklearn.model_selection import train_test_split


def load_data(data_ratio = 0.02, test_ratio = 0.3):
    '''
    load trade data, including fraud trades and no_fraud trades, due to the unbalance of fraud trades and no_fraud
    trades, we use all fraud trades and choose some no_fraud trades to train our model, when we calculate the accuracy
    of our model, we must consider the data ratio
    Args:
        data_ratio: the ratio to choose no_fraud trades
        test_ratio: the ratio for test data using in our models
    Returns:
        X_train: features used in model training
        y_train: labels used in model training
        X_test: features used in model testing
        y_test: labels used in model testing
    '''
    feature = []
    label = []
    file_read = open('data/fraud_trade.txt')  # read fraud trades
    for line in file_read.readlines():
        data = line.strip().split()
        label.append(int(data[0]))  # the first column is the label, data type: int
        feature0 = [float(item) for item in data[1:]]  # from the second column to the end are the features, data type: float
        feature.append(feature0)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(feature, label, test_size = test_ratio, random_state = 0)
    feature = []
    label = []
    sample = int(1.0/data_ratio)  # sample number, label the trade to choose or not
    count = 0  # count number
    file_read = open('data/no_fraud_trade.txt')  # read no_fraud trades
    logger.info("load file")
    for line in file_read.readlines():
        if count % sample == 0:
            data = line.strip().split()
            label.append(int(data[0]))  # the first column is the label, data type: int
            feature0 = [float(item) for item in data[1:]]  # from the second column to the end are the features, data type: float
            feature.append(feature0)
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=test_ratio, random_state=0)
    X_train = X_train + X_train1  # connect the features and labels together
    y_train = y_train + y_train1
    X_test = X_test + X_test1
    y_test = y_test + y_test1
    return X_train, X_test, y_train, y_test
