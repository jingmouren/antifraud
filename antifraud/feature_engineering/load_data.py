
def load_train_test(train_file, test_file):
    """
    load data from train and test files out of the project
    Args:
        train_file: a string of train data address
        test_file: a string of test data address
    Returns:
        train_feature: none
        train_label: none
        test_feature: none
        test_label: none
    """
    # load train data from train file
    train_feature = []
    train_label = []
    file_read = open(train_file)
    for line in file_read.readlines():
        data = line.strip().split()
        train_label.append(int(data[0]))  # the first column is the label, data type: int
        # from the second column to the end are the features, data type: float
        _feature = [float(item) for item in data[1:]]
        train_feature.append(_feature)
    # load test data from test file
    test_feature = []
    test_label = []
    file_read = open(test_file)
    for line in file_read.readlines():
        data = line.strip().split()
        test_label.append(int(data[0]))  # the first column is the label, data type: int
        # from the second column to the end are the features, data type: float
        _feature = [float(item) for item in data[1:]]
        test_feature.append(_feature)
    return train_feature, train_label, test_feature, test_label


