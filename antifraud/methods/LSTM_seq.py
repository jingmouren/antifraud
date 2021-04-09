from keras.layers import Dense, LSTM
from keras.models import Sequential
import numpy as np


def LSTM_model(train_feature_dir, train_label_dir, test_feature_dir, test_label_dir):
    """read data"""
    train_feature = np.load(train_feature_dir)
    train_label = np.load(train_label_dir)
    test_feature = np.load(test_feature_dir)
    test_label = np.load(test_label_dir)
    """ data reshape"""
    train_feature = np.reshape(train_feature, (len(train_feature), 1, 1, 3584))
    test_feature = np.reshape(test_feature, (len(test_feature), 1, 1, 3584))
    new_train_feature = []
    for i in range(len(train_feature)):
        new_train_feature.append(train_feature[i][0])
    new_test_feature = []
    for i in range(len(test_feature)):
        new_test_feature.append(test_feature[i][0])

    # parameters for LSTM
    nb_lstm_outputs = 30
    nb_time_steps = 1
    nb_input_vector = 3584

    # build model
    model = Sequential()
    model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
    model.add(Dense(1, activation='softmax'))

    # compile:loss, optimizer, metrics
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(np.array(new_train_feature), train_label, epochs=2, batch_size=128, verbose=1)

    pre_label = model.predict_proba(np.array(new_test_feature))
    _pre_label = []
    for i in range(len(test_feature)):
        _pre_label.append(pre_label[i][1])
    return np.array(_pre_label), test_label

