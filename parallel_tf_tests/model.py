"""
Defines a simple FFNN
"""

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf

class FFNN(object):
    """
    Class representation of a neural net.
    Basically a struct used to hold some Tensors in a TF graph.
    """

    def __init__(self, input_dimension, output_dimension, num_layers, width, activation, graph):
        """
        :param input_dimension: the number of inputs into the first layer
        :type input_dimension: int
        :param output_dimension: the number of outputs from the last layer
        :type output_dimension: int
        :param num_layers: the number of layers in this MLP
        :type num_layers: int
        :param width: the number of neurons in each hiden layer
        :type width: int
        :param activation: the activation function each layer uses
        :type activation: Tensor -> Tensor
        :param graph: the graph we will be adding operations to
        :type graph: tf.Graph
        :raises: ValueError
        """
        with graph.as_default():
            # input checking
            if input_dimension <= 0 or output_dimension <= 0:
                raise ValueError('Input and output dimensions must be positive')
            if num_layers <= 0:
                raise ValueError('Must have a at least one layer')
            if width <= 0:
                raise ValueError('Must have at least one neuron in each layer')

            self.inputs = tf.placeholder(tf.float32, shape=(None, input_dimension))
            self.targets = tf.placeholder(tf.float32, shape=(None, output_dimension))

            # build hidden layers
            def build_layer(x, input_size, output_size, layer_index, use_activation=True):
                """
                :param x: input to layer
                :param input_size: the number of dimensions of the input
                :param output_size: number of dimensions of the output
                :param layer_index: which layer are we in the network?
                """
                with tf.name_scope('layer{0}'.format(layer_index)):
                    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.1))
                    b = tf.Variable(tf.zeros([output_size]))
                    y = tf.matmul(x, W) + b

                    if use_activation:
                        y = activation(y)

                    return (y, output_size)

            # add first layer
            self._layers = [build_layer(self.inputs, input_dimension, output_dimension, 0)]
            # add other hidden layers
            for i in range(num_layers):
                self._layers.append(build_layer(self._layers[-1][0], self._layers[-1][1], width, i + 1))
            # add output layer
            output_layer = build_layer(self._layers[-1][0], width, output_dimension, num_layers+1, False)
            self._layers.append(output_layer)

            self.inference = tf.nn.softmax(self._layers[-1][0])

            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self._layers[-1][0])
            self.loss = tf.reduce_mean(self.loss)

            self.training = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def train(self, x, targets, sess):
        """
        Perform one step of training using the provided input and targets in the provided session.
        """
        feed_dict = {self.inputs: x, self.targets: targets}

        sess.run(self.training, feed_dict)

    def get_loss(self, x, targets, sess):
        """
        Get the loss
        """
        feed_dict = {self.inputs: x, self.targets: targets}

        return sess.run(self.loss, feed_dict)
