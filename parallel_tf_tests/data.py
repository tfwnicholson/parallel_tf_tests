"""
Ubiquotous data module.

A lighweight wrapper around read_data_sets for MNIST that yields a requested sized subset.
"""

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# where we store the data, just dump it here and leave it lying around
DATA_DIR = 'data'
"""
Where we cache the full data set.
"""

def get_mnist(num_examples=None):
    """
    Grab specified number of MNIST input output pairs from the training data.
    :param num_examples: the number of examples to get. If None, then all examples are returned
    :type num_examples: int
    :returns: equal length pairs corresponding to input images and output labels
    :rtype: (list(np.ndarray), list(numpy.uint8))
    """
    data_sets = read_data_sets(DATA_DIR)

    if num_examples is None or num_examples > data_sets.train.num_examples:
        # give you the maximum number from the training examples
        num_examples = data_sets.train.num_examples
    elif num_examples <= 0:
        raise ValueError('Need a sensible length mate')
    

    cut_data_set = list(zip(data_sets.train.images, data_sets.train.labels))[:num_examples]

    return list(zip(*cut_data_set))
