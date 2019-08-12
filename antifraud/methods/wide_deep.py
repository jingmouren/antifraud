from __future__ import print_function
import tensorflow as tf
import pandas as pd
import warnings
from antifraud.__main__ import parse_args
warnings.filterwarnings("ignore")


def wide_deep_load_data(file):
    # Define the column names for the data sets.
    columns = ["label", "match", "county", "amt_g", "amt_p", "trade_time", "trade_date"]
    # Read the training or test data sets into Pandas dataframe.
    df = pd.read_csv(file, sep=' ', dtype=str, names=columns, skipinitialspace=True)
    df['label'] = df['label'].astype(int)
    df['amt_g'] = df['amt_g'].astype(float)
    df['amt_p'] = df['amt_p'].astype(float)
    return df


def input_fn(flag):
    args = parse_args()
    if flag == 'train':
        df = wide_deep_load_data(args.train)
    if flag == 'test':
        df = wide_deep_load_data(args.test)
    label_columns = 'label'
    categorical_columns = ["match", "county", "trade_time", "trade_date"]
    continuous_columns = ["amt_g", "amt_p"]
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in continuous_columns}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in categorical_columns}
    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[label_columns].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    flag = 'train'
    return input_fn(flag)


def eval_input_fn():
    flag = 'test'
    return input_fn(flag)


def wide_deep():
    # Categorical base columns.
    match = tf.contrib.layers.sparse_column_with_hash_bucket("match", hash_bucket_size=20000)
    county = tf.contrib.layers.sparse_column_with_hash_bucket("county", hash_bucket_size=1000)
    trade_time = tf.contrib.layers.sparse_column_with_hash_bucket("trade_time", hash_bucket_size=100)
    trade_date = tf.contrib.layers.sparse_column_with_hash_bucket("trade_date", hash_bucket_size=100)

    # Continuous base columns.
    amt_g = tf.contrib.layers.real_valued_column("amt_g")
    amt_p = tf.contrib.layers.real_valued_column("amt_p")

    wide_columns = [
        match, county, trade_time, trade_date,
        tf.contrib.layers.crossed_column([match, county], hash_bucket_size=int(1e4)),
        tf.contrib.layers.crossed_column([match, county, trade_time], hash_bucket_size=int(1e4)),
        tf.contrib.layers.crossed_column([match, county, trade_date], hash_bucket_size=int(1e6))]

    deep_columns = [
        tf.contrib.layers.embedding_column(match, dimension=8),
        tf.contrib.layers.embedding_column(county, dimension=8),
        tf.contrib.layers.embedding_column(trade_time, dimension=8),
        tf.contrib.layers.embedding_column(trade_date, dimension=8),
        amt_g, amt_p]
    # set the model
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir='model',
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
    # fit the model
    m.fit(input_fn=train_input_fn, steps=200)
    results = m.evaluate(input_fn=eval_input_fn, steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))





