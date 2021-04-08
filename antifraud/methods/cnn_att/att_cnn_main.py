import numpy as np
import datetime
import tensorflow as tf
from antifraud.config import Config
from antifraud.methods.cnn_att.att_cnn_2d import Att_cnn2d_model
from antifraud.methods.cnn_att.att_cnn_3d import Att_cnn3d_model
att_config = Config()

def make_loss_op(target, output):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
    return tf.reduce_mean(losses)

def make_accuracy_op(target, output):
    correct_predictions = tf.equal(output, tf.argmax(target, 1))
    return tf.reduce_mean(tf.cast(correct_predictions, "float"))



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def att_train_2d(x_train, y_train, x_test, y_test, att_config=att_config):
    tf.reset_default_graph()
    model = Att_cnn2d_model()
    input_x, input_y = model.create_placeholder(time_window_length=att_config.input_shape_2d[0],
                                                measure_length=att_config.input_shape_2d[1],
                                                num_classes=att_config.num_classes)
    input_x = model.attention_layer(inputs=input_x,
                                    attention_hidden_dim=att_config.attention_hidden_dim,
                                    time_window_length=att_config.input_shape_2d[0],
                                    measure_length=att_config.input_shape_2d[1])
    input_x = model.cnn_layers(inputs=input_x,
                               filter_sizes=att_config.filter_sizes,
                               num_filters=att_config.num_filters)
    output_scores, predictions = model.dense_layers(input_x,
                                                    num_classes=att_config.num_classes)
    output_probabilities = tf.nn.sigmoid(output_scores, name="probabilities")
    print("Constructing optimizer operation...")
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    loss_op = make_loss_op(input_y, output_scores)
    accuracy_op = make_accuracy_op(input_y, predictions)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    grads_and_vars = optimizer.compute_gradients(loss_op)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                         global_step=global_step)
    print("[ Att_cnn2d_model ] Done! Start training now.")
    try:
        sess.close()
    except NameError:
        pass
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if len(y_train.shape) == 1:
        from keras.utils import to_categorical
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    batches = batch_iter(list(zip(x_train, y_train)),
                         batch_size=att_config.batch_size,
                         num_epochs=att_config.num_epochs)
    del x_train, y_train

    test_feed_dict = {
        model.input_x: x_test,
        model.input_y: y_test
    }

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch
        }
        train_values = sess.run({
            "train": train_op,
            "step": global_step,
            "loss": loss_op,
            "accuracy": accuracy_op
        },
        feed_dict=feed_dict)
        current_step = tf.train.global_step(sess, global_step)
        time_str = datetime.datetime.now().isoformat()
        if current_step % att_config.evaluate_every == 0:
            test_values = sess.run({
                "step": global_step,
                "loss": loss_op,
                "accuracy": accuracy_op
                },
                feed_dict=test_feed_dict)
            print("{}: step {}, train_loss {:.5f}, train_acc {:.5f}, test_loss {:.5f}, test_acc {:.5f}".format(time_str, train_values['step'], train_values['loss'], train_values['accuracy'], test_values['loss'], test_values['accuracy']))

    Evaluation_values = sess.run({
        "scores": output_probabilities,
        "loss": loss_op
        },
        feed_dict=test_feed_dict)

    y_score = Evaluation_values["scores"]
    return y_score

def att_train_3d(x_train, y_train, x_test, y_test, att_config=att_config):
    tf.reset_default_graph()
    model = Att_cnn3d_model()
    input_x, input_y = model.create_placeholder(time_window_length=att_config.input_shape_3d[0],
                                                space_window_length=att_config.input_shape_3d[1],
                                                measure_length=att_config.input_shape_3d[2],
                                                num_classes=att_config.num_classes)
    input_x = model.attention_layer(inputs=input_x,
                                    attention_hidden_dim=att_config.attention_hidden_dim,
                                    time_window_length=att_config.input_shape_3d[0],
                                    space_window_length=att_config.input_shape_3d[1],
                                    measure_length=att_config.input_shape_3d[2],)
    input_x = model.cnn_layers(inputs=input_x,
                               filter_sizes=att_config.filter_sizes,
                               num_filters=att_config.num_filters)
    output_scores, predictions = model.dense_layers(input_x,
                                                    num_classes=att_config.num_classes)
    output_probabilities = tf.nn.sigmoid(output_scores, name="probabilities")
    print("Constructing optimizer operation...")
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    loss_op = make_loss_op(input_y, output_scores)
    accuracy_op = make_accuracy_op(input_y, predictions)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    grads_and_vars = optimizer.compute_gradients(loss_op)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                         global_step=global_step)
    print("[ Att_cnn3d_model ] Done! Start training now.")

    try:
        sess.close()
    except NameError:
        pass
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if len(y_train.shape) == 1:
        from keras.utils import to_categorical
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    batches = batch_iter(list(zip(x_train, y_train)),
                         batch_size=att_config.batch_size,
                         num_epochs=att_config.num_epochs)
    #del x_train, y_train

    test_feed_dict = {
        model.input_x: x_test,
        model.input_y: y_test
    }

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch
        }
        train_values = sess.run({
            "train": train_op,
            "step": global_step,
            "loss": loss_op,
            "accuracy": accuracy_op
        },
        feed_dict=feed_dict)
        current_step = tf.train.global_step(sess, global_step)
        time_str = datetime.datetime.now().isoformat()
        if current_step % att_config.evaluate_every == 0:
            test_values = sess.run({
                "step": global_step,
                "loss": loss_op,
                "accuracy": accuracy_op
                },
                feed_dict=test_feed_dict)
            print("{}: step {}, train_loss {:.5f}, train_acc {:.5f}, test_loss {:.5f}, test_acc {:.5f}".format(time_str, train_values['step'], train_values['loss'], train_values['accuracy'], test_values['loss'], test_values['accuracy']))

    Evaluation_values = sess.run({
        "scores": output_probabilities,
        "loss": loss_op
        },
        feed_dict=test_feed_dict)

    y_score = Evaluation_values["scores"]
    return y_score


def att_main(x_train, y_train, x_test, y_test, mode):
    y_pred = np.zeros(shape=y_test.shape)
    if mode == "cnn-att-2d":
        y_pred = att_train_2d(x_train,y_train,x_test,y_test)
    elif mode == "cnn-att-3d":
        y_pred = att_train_3d(x_train,y_train,x_test,y_test)

    return y_pred

def load_att_data(train_feature="train_feature.npy", train_label="train_label.npy", test_feature="test_feature.npy", test_label="test_label.npy"):
    x_train = np.load(train_feature)
    print("shape of x_train: {}".format(x_train.shape))
    y_train = np.load(train_label)
    x_test = np.load(test_feature)
    y_test = np.load(test_label)
    return x_train, y_train, x_test, y_test
