# From Edward (edward/edward/util/graphs.py) (https://github.com/blei-lab/edward)
import numpy as np
import random
import tensorflow as tf
import six

def set_seed(x):
    """Set seed for both NumPy and TensorFlow.
    Args:
        x: int, float.
        seed
    """
    node_names = list(six.iterkeys(tf.get_default_graph()._nodes_by_name))
    if len(node_names) > 0 and node_names != ['keras_learning_phase']:
        raise RuntimeError("Seeding is not supported after initializing "
                           "part of the graph. "
                           "Please move set_seed to the beginning of your code.")
    random.seed(x)
    np.random.seed(x)
    tf.set_random_seed(x)

def printing(var_tf):
    return var_tf
