"""
Test the data module.
"""

import unittest

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import parallel_tf_tests.data


class TestMNISTData(unittest.TestCase):
    """
    Test we can load the right number of data items
    """
    def test_right_number(self):
        lengths_to_assert = [10, 100, 1000, 10000]
        for length in lengths_to_assert:
            x, y = parallel_tf_tests.data.get_mnist(length)
            self.assertEqual(length, len(x))
            self.assertEqual(length, len(y))

    def test_wrong_number(self):
        lengths_to_assert = [-1000, -1, 0]
        for length in lengths_to_assert:
            with self.assertRaises(ValueError):
                parallel_tf_tests.data.get_mnist(length)

    def test_cap_length(self):
        num_training_examples = read_data_sets(parallel_tf_tests.data.DATA_DIR).train.num_examples
        lengths_to_assert = [num_training_examples + 1, num_training_examples + 2, None]
        for length in lengths_to_assert:
            x, y = parallel_tf_tests.data.get_mnist(length)
            self.assertEqual(len(x), num_training_examples)
            self.assertEqual(len(y), num_training_examples)
