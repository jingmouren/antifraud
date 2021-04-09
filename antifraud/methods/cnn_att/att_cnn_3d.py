import tensorflow as tf
import numpy as np


class Att_cnn3d_model():
    '''
    An embedding layer named Attention layer,
    connected by cnn model, which is composed by
    convolution, max-pooling and dense layers.
    '''
    def __init__(self, **kwargs):
        print("[ Att_cnn3d_model ] Start Initializing...")
        self.kwargs = kwargs
        self.lamda = 0.0
        tmp_name = "lamda"
        if tmp_name in kwargs:
            self.lamda = float(kwargs[tmp_name])
        # Written by Sheng.Xiang
        print("[ Att_cnn3d_model ] Initialization finished, Waiting for placeholders...")


    def create_placeholder(self, time_window_length, space_window_length, measure_length, num_classes):
        '''
        Initialize tensorflow placeholders for input features and target labels.
        '''
        self.input_x = tf.placeholder(tf.float32,
                                 shape=[None, time_window_length, space_window_length, measure_length],
                                 name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                 shape=[None, num_classes],
                                 name="input_y")
        return self.input_x, self.input_y

    def attention_layer(self, inputs, attention_hidden_dim=100, **kwargs):
        print("[ Att_cnn3d_model ] Constructing Attention layer... ")
        self.attention_hidden_dim = attention_hidden_dim
        self.measure_length = int(kwargs['measure_length'])
        self.time_window_length = int(kwargs['time_window_length'])
        self.space_window_length = int(kwargs['space_window_length'])
        self.embedded_maps = inputs
        # Initialize weights matrix
        # Wa = [attention_W  attention_U]
        self.attention_W1 = tf.Variable(
            tf.random_uniform([self.measure_length*self.space_window_length,
                               self.attention_hidden_dim], 0.0, 1.0),
            name="attention_W1")
        self.attention_W2 = tf.Variable(
            tf.random_uniform([self.measure_length*self.time_window_length,
                               self.attention_hidden_dim], 0.0, 1.0),
            name="attention_W2")
        self.attention_V1 = tf.Variable(
            tf.random_uniform([self.attention_hidden_dim, self.time_window_length], 0.0, 1.0),
            name="attention_V1")
        self.attention_V2 = tf.Variable(
            tf.random_uniform([self.attention_hidden_dim, self.space_window_length], 0.0, 1.0),
            name="attention_V2")
        # Attention Layer before Convolution
        self.output_att = []
        print("shape of inputs: {}".format(inputs.shape))
        with tf.name_scope("attention-time-space"):
            # Time & Space attention
            self.output_att.append(self.attention(self.embedded_maps))

            input_conv = tf.reshape(self.output_att,[-1, self.time_window_length, self.space_window_length, self.measure_length])
            self.input_conv_expanded = tf.expand_dims(input_conv, -1)
        return self.input_conv_expanded

    def cnn_layers(self, inputs, **kwargs):
        print("[ Att_cnn3d_model ] Constructing Convolution layers... ")
        if len(inputs.shape) == 4:
            self.input_conv_expanded = tf.expand_dims(inputs, -1)
        elif len(inputs.shape) == 5:
            self.input_conv_expanded = inputs
        else:
            print("Wrong conv input shape!")

        filter_sizes = kwargs['filter_sizes']
        num_filters = kwargs['num_filters']
        for i, filter_size in enumerate(zip(filter_sizes, num_filters)):
            with tf.name_scope("conv-maxpool-%s" % filter_size[0]):
                channels_input_conv = self.input_conv_expanded.shape[-1].value
                # Convolution Layer
                filter_shape = [filter_size[0], filter_size[0], filter_size[0], channels_input_conv, filter_size[1]]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_size[1]]), name="b")
                conv = tf.nn.conv3d(
                    self.input_conv_expanded,
                    W,
                    strides=[1, 1, 1, 1, 1],
                    padding="VALID",
                    name="convolution-%s" % filter_size[0])
                # Activation
                self.input_conv_expanded = tf.nn.relu(tf.nn.bias_add(conv, b), name="RelU")
        return self.input_conv_expanded

    def dense_layers(self, inputs, num_classes, **kwargs):
        print("[ Att_cnn3d_model ] Constructing full connected layers... ")
        num_neuron_total = 1
        for index, item in enumerate(inputs.shape):
            if index != 0:
                num_neuron_total *= item

        self.h_flat = tf.reshape(inputs, [-1, num_neuron_total])

        with tf.name_scope("dense-256"):
            W256 = tf.get_variable(
                "W256",
                shape=[num_neuron_total, 256],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b256 = tf.Variable(tf.constant(0.1, shape=[256]), name="b256")
            self.scores = tf.nn.xw_plus_b(self.h_flat, W256, b256, name="scores-256")
            self.h_flat = tf.nn.sigmoid(self.scores, name="probabilities-256")

        with tf.name_scope("dense-32"):
            W32 = tf.get_variable(
                "W32",
                shape=[256, 32],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b32 = tf.Variable(tf.constant(0.1, shape=[32]), name="b32")
            self.scores = tf.nn.xw_plus_b(self.h_flat, W32, b32, name="scores-32")
            self.h_flat = tf.nn.sigmoid(self.scores, name="probabilities-32")

        # Final (unscaled) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[32, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.scores = tf.nn.xw_plus_b(self.h_flat, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        print("[ Att_cnn3d_model ] Model building finished ")
        return self.scores, self.predictions

    def attention(self, x):
        """
        Attention model
        :param x: the embedded input of all times and places(x_ij of attentions)
                    X_all = X{Time,Space,Measures}
        :param index: step of time
        """
        # time attention
        e_i = []
        print("shape of inputs: {}".format(x.shape))
        x = tf.split(x, self.time_window_length, axis=1)
        for xi in range(len(x)):
            x_i = tf.reshape(x[xi],[-1,self.space_window_length*self.measure_length])
            atten_hidden_t = tf.nn.relu(tf.matmul(x_i, self.attention_W1))
            e_i_j = tf.matmul(atten_hidden_t, self.attention_V1)
            #print("shape of alpha_i_j: {}".format(e_i_j.shape))
            e_i.append([tf.nn.softmax(e_i_j*(1-self.lamda))])

        e_i = tf.concat(e_i, axis=0)
        t_i = []
        alpha_i = tf.reduce_sum(tf.transpose(e_i), 0)
        #alpha_i = tf.split(alpha_i,self.time_window_length,axis=0)
        print("shape of alpha_i: {}".format(alpha_i.shape))
        #print("shape of e_i: {}".format(e_i.shape))

        for xi, x_i in enumerate(x):
            #print("shape of x_i: {}".format(x_i.shape))
            x_i = tf.reshape(x[xi],[-1,self.space_window_length*self.measure_length])
            #print("shape of x_i: {}".format(x_i.shape))
            x_i = tf.matmul(tf.matrix_diag(alpha_i[:,xi]), x_i)
            #print("shape of alpha_i_j: {}".format(alpha_i[xi].shape))
            t_i.append([tf.reshape(x_i,[-1,self.space_window_length,self.measure_length])])

        # space attention
        t_i = tf.concat(t_i,axis=0)
        print("shape of t_i: {}".format(t_i.shape))
        x = tf.transpose(t_i,[1,2,0,3])
        print("shape of inputs: {}".format(x.shape))
        f_i = []
        x = tf.split(x, self.space_window_length, axis=1)
        for xi in range(len(x)):
            x_i = tf.reshape(x[xi],[-1,self.time_window_length*self.measure_length])
            atten_hidden_s = tf.nn.relu(tf.matmul(x_i, self.attention_W2))
            f_i_j = tf.squeeze(tf.matmul(atten_hidden_s, self.attention_V2))
            f_i.append(tf.nn.softmax(f_i_j*(1-self.lamda)))

        s_i = []
        alpha_j = tf.reduce_sum(tf.transpose(f_i), 0)
        for xi in range(len(x)):
            x_i = tf.reshape(x[xi],[-1,self.time_window_length*self.measure_length])
            x_i = tf.matmul(tf.matrix_diag(alpha_i[:,xi]), x_i)
            s_i.append([tf.reshape(x_i,[-1,self.time_window_length,self.measure_length])])

        s_i = tf.concat(s_i,axis=0)
        x = tf.transpose(s_i,[1,2,0,3],name="input_conv")
        print("shape of inputs: {}".format(x.shape))
        return x

    def time_attention(self, x_i, x):
        if not isinstance(x, list):
            x = tf.split(x, self.time_window_length, axis=1)

        for i in range(len(x)):
            output = x[i]
            output = tf.reshape(output, [-1, self.space_window_length*self.measure_length])
            atten_hidden_t = tf.nn.relu(tf.matmul(x_i, self.attention_W1))
            e_i_j = tf.squeeze(tf.matmul(atten_hidden_t, self.attention_V1))


