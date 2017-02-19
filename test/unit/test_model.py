"""
Simple unit tests for the FFNN model
"""
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import unittest

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import parallel_tf_tests.model

class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mnist = input_data.read_data_sets("data", one_hot=True)

    def create_nn(self, num_layers, width, activation, graph):
        return parallel_tf_tests.model.FFNN(784, 10, num_layers, width, activation, graph)

    def test_tiny(self):
        """Test a really small (784 x 4 x10) sigmoid net"""
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        nn = self.create_nn(1, 4, tf.nn.sigmoid, graph)
        
        data_in, data_out = self.mnist.train.next_batch(100)
        with sess as sess:
            nn.init(sess)
            
            # get initial loss
            loss_before = nn.get_loss(data_in, data_out, sess)
            # train a few times:
            for _ in range(100):
                nn.train(data_in, data_out, sess)
            loss_after = nn.get_loss(data_in, data_out, sess)
            self.assertLess(loss_after, loss_before)


    def test_medium(self):
        """Test a really small (784 x 4 x10) sigmoid net"""
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        nn = self.create_nn(4, 200, tf.nn.sigmoid, graph)
        
        data_in, data_out = self.mnist.train.next_batch(100)

        with sess as sess:
            nn.init(sess)

            # get initial loss
            loss_before = nn.get_loss(data_in, data_out, sess)
            # train a few times:
            for _ in range(100):
                nn.train(data_in, data_out, sess)
            loss_after = nn.get_loss(data_in, data_out, sess)
            self.assertLess(loss_after, loss_before)
        
