from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np


def cnn_model(train_feature_dir, train_label_dir, test_feature_dir, test_label_dir):
    """read data"""
    train_feature = np.load(train_feature_dir)
    train_label_single = np.load(train_label_dir)
    train_label = to_categorical(train_label_single)
    test_feature = np.load(test_feature_dir)
    test_label_single = np.load(test_label_dir)
    test_label = to_categorical(test_label_single)
    """ data reshape"""
    train_feature = np.reshape(train_feature, (len(train_feature), 512, 7, 1))
    test_feature = np.reshape(test_feature, (len(test_feature), 512, 7, 1))
    """build model"""
    cnn_max = Sequential()
    cnn_max.add(Conv2D(32, (2, 2), activation='relu', input_shape=(512, 7, 1)))
    cnn_max.add(Conv2D(64, (2, 2)))
    cnn_max.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_max.add(Flatten())
    cnn_max.add(Dense(128, activation='relu'))
    cnn_max.add(Dense(64, activation='relu'))
    cnn_max.add(Dense(2, activation='softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    cnn_max.compile(loss='categorical_crossentropy', optimizer=sgd)
    """train the model"""
    cnn_max.fit(train_feature, train_label, batch_size=32, epochs=10)
    """test the model"""
    pre_label = cnn_max.predict_proba(test_feature)
    _pre_label = []
    for i in range(len(test_feature)):
        _pre_label.append(pre_label[i][1])
    return np.array(_pre_label), test_label_single

