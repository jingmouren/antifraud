import tensorflow as tf
import numpy as np


class Att_cnn2d_model():
    '''
    An embedding layer named Attention layer,
    connected by cnn model, which is composed by
    convolution, max-pooling and dense layers.
    '''
    def __init__(self, **kwargs):
        print("[ Att_cnn2d_model ] Start Initializing...")
        self.kwargs = kwargs
        self.lamda = 0.0
        print("[ Att_cnn2d_model ] Initialization finished, Waiting for placeholders...")


    def create_placeholder(self, time_window_length, measure_length, num_classes):
        '''
        Initialize tensorflow placeholders for input features and target labels.
        '''
        self.input_x = tf.placeholder(tf.float32,
                                 shape=[None, time_window_length, measure_length],
                                 name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                 shape=[None, num_classes],
                                 name="input_y")
        return self.input_x, self.input_y

    def attention_layer(self, inputs, attention_hidden_dim=100, **kwargs):
        print("[ Att_cnn2d_model ] Constructing Attention layer... ")
        self.attention_hidden_dim = attention_hidden_dim
        self.measure_length = int(kwargs['measure_length'])
        self.time_window_length = int(kwargs['time_window_length'])
        self.embedded_maps = inputs
        # Initialize weights matrix
        # Wa = [attention_W  attention_U]
        self.attention_W = tf.Variable(
            tf.random_uniform([self.measure_length, self.attention_hidden_dim], 0.0, 1.0),
            name="attention_W")
        self.attention_U = tf.Variable(
            tf.random_uniform([self.measure_length, self.attention_hidden_dim], 0.0, 1.0),
            name="attention_U")
        self.attention_V = tf.Variable(
            tf.random_uniform([self.attention_hidden_dim, 1], 0.0, 1.0),
            name="attention_V")
        # Attention Layer before Convolution
        self.output_att = []
        with tf.name_scope("attention"):
            input_att = tf.split(self.embedded_maps, self.time_window_length, axis=1)
            for index, x_i in enumerate(input_att):
                #print("shape of x_i: {}".format(x_i.shape))
                x_i = tf.reshape(x_i, [-1, self.measure_length])
                c_i = self.attention(x_i, input_att, index)
                # combine two feature maps to one map
                print("shape of x_i: {}".format(x_i.shape))
                print("shape of c_i: {}".format(c_i.shape))
                inp = tf.concat([x_i, c_i], axis=1)
                #inp = c_i
                self.output_att.append(inp)

            input_conv = tf.reshape(tf.concat(self.output_att, axis=1),
                                    [-1, self.time_window_length, self.measure_length*2],
                                    name="input_convolution")
            # 最后一个维度加上，适应cnn模型
            self.input_conv_expanded = tf.expand_dims(input_conv, -1)
        return self.input_conv_expanded

    def cnn_layers(self, inputs, **kwargs):
        print("[ Att_cnn2d_model ] Constructing Convolution layers... ")
        if len(inputs.shape) == 3: # 检查维度是否符合2d卷积层的输入
            self.input_conv_expanded = tf.expand_dims(inputs, -1)
        elif len(inputs.shape) == 4:
            self.input_conv_expanded = inputs
        else:
            print("Wrong conv input shape!")

        filter_sizes = kwargs['filter_sizes']
        num_filters = kwargs['num_filters']
        for i, filter_size in enumerate(zip(filter_sizes, num_filters)):
            with tf.name_scope("conv-maxpool-%s" % filter_size[0]):
                channels_input_conv = self.input_conv_expanded.shape[-1].value
                # Convolution Layer
                filter_shape = [filter_size[0], filter_size[0], channels_input_conv, filter_size[1]]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_size[1]]), name="b")
                conv = tf.nn.conv2d(
                    self.input_conv_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="convolution-%s" % filter_size[0])
                # Activation
                self.input_conv_expanded = tf.nn.relu(tf.nn.bias_add(conv, b), name="RelU")
                # Max Pooling(abandoned)
                #pooled = tf.nn.max_pool(
                #    h,
                #    ksize=[1, time_window_length - filter_size[0] + 1, 1, 1],
                #    strides=[1, 1, 1, 1],
                #    padding='VALID',
                #    name="pool")
        return self.input_conv_expanded

    def dense_layers(self, inputs, num_classes, **kwargs):
        print("[ Att_cnn2d_model ] Constructing full connected layers... ")
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
            #l2_loss += tf.nn.l2_loss(W)
            #l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_flat, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        print("[ Att_cnn2d_model ] Model building finished ")
        return self.scores, self.predictions

    def attention(self, x_i, x, index):
        """
        Attention model for Neural Machine Translation
        :param x_i: the embedded input at time i
        :param x: the embedded input of all times(x_j of attentions)
        :param index: step of time
        """

        e_i = []
        c_i = []
        for i in range(len(x)):
            output = x[i]
            output = tf.reshape(output, [-1, self.measure_length])
            #两个权重矩阵是否需要学习？？
            atten_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
            #atten_hidden = tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j*tf.pow(1-self.lamda,abs(i-index)))
        e_i = tf.concat(e_i, axis=1)
        # e_i = tf.exp(e_i)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.time_window_length, 1)
        #alpha_i = tf.split(alpha_i, self.measure_length, 1)

        # i!=j
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, self.measure_length])
                c_i_j = tf.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.time_window_length-1, self.measure_length])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i

