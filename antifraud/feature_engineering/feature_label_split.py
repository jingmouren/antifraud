import numpy as np


def split_label_feature(file_dir):
    array = np.load(file=file_dir)
    length = len(array)
    feature = []
    label = []
    for i in range(length):
        label.append(array[i][0])
        feature.append(array[i][1])
    feature = np.array(feature)
    label = np.array(label)
    np.save(file='train_feature'+file_dir, arr=feature)
    np.save(file="train_label"+file_dir, arr=label)
    return 0



