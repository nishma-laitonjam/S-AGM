# Writing a numpy array as tf.records
# To create the tfrecord for the SGRLD

import numpy as np
import tensorflow as tf
import random
from scipy import sparse
import timeit


def rand_sample(n, M):
    return np.random.choice(n, M)

def np_to_tfrecords1(full_b, full_b_test, num_of_samples, L, Ms):

    def _dtype_feature(ndarray):
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float32 or dtype_ == np.float64:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        elif dtype_ == np.uint8:
            return lambda array: tf.train.Feature(bytes_list=tf.train.BytesList(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. Instead got {}".format(ndarray.dtype))

    n = full_b.shape[0]

    dtype_feature_value = _dtype_feature(np.array([[1]]))


    # iterate over each sample, and serailize it as ProtoBuf.
    for idx in range(num_of_samples):
        d_feature = {}
        curr_node = np.array(random.sample(range(n), 1))
        if random.uniform(0, 1) > 0.5:
            d_feature['edge'] = dtype_feature_value([curr_node, 1])  # int64
            train_edge = sparse.find(full_b[curr_node, :])[1]
            global_mini_batch_indices = train_edge
        else:
            d_feature['edge'] = dtype_feature_value([curr_node, 0])  # int64
            # train_non_edge = sparse.find((full_b[curr_node, :] + full_b_test[curr_node, :]) == 0)[1]
            curr_row = full_b[curr_node, :] + full_b_test[curr_node, :]
            train_non_edge = np.where(curr_row.todense() == 0)[1]
            global_mini_batch_indices = random.sample(list(train_non_edge), len(train_non_edge))[0:L]  # added list to be compatible with 3.7 and 2.7
        ########################## Common ##########################


        d_feature['global_mini_batch_indices'] = dtype_feature_value(global_mini_batch_indices)  # int64
        mini_batch_size = len(global_mini_batch_indices)
        n_rep = np.repeat(n, mini_batch_size + 1)
        M_rep = np.repeat(M, mini_batch_size + 1)
        local_mini_batch_indices_m = np.array(list(map(rand_sample, n_rep, M_rep)))  # use list to be compatible with python 3.7 and 2.7
        d_feature['local_mini_batch_indices_m'] = dtype_feature_value(local_mini_batch_indices_m.reshape(-1))  # int64
        curr_indices = np.append(global_mini_batch_indices, [curr_node])


        row_train_indices = full_b[curr_indices].toarray()
        row_test_indices = full_b_test[curr_indices].toarray()
        row_index = np.array(range(mini_batch_size + 1))
        w_train_indices = row_train_indices[row_index[:, np.newaxis], local_mini_batch_indices_m]
        w_test_diag_indices = row_test_indices[row_index[:, np.newaxis], local_mini_batch_indices_m]


        curr_indices_train_indices = np.array(np.where(w_train_indices))
        if curr_indices_train_indices.size == 0:
            d_feature['curr_indices_train_indices_0'] = dtype_feature_value([])  # int64
            d_feature['curr_indices_train_indices_1'] = dtype_feature_value([])  # int64
            d_feature['true_curr_indices_train_indices_0'] = dtype_feature_value([])  # int64
            d_feature['true_curr_indices_train_indices_1'] = dtype_feature_value([])  # int64
        else:
            d_feature['curr_indices_train_indices_0'] = dtype_feature_value(curr_indices_train_indices[0])  # int64
            d_feature['curr_indices_train_indices_1'] = dtype_feature_value(curr_indices_train_indices[1])  # int64
            true_curr_indices_train_indices_0 = curr_indices[curr_indices_train_indices[0]]
            d_feature['true_curr_indices_train_indices_0'] = dtype_feature_value(true_curr_indices_train_indices_0)  # int64
            true_curr_indices_train_indices_1 = local_mini_batch_indices_m[np.where(w_train_indices)]
            d_feature['true_curr_indices_train_indices_1'] = dtype_feature_value(true_curr_indices_train_indices_1)  # int64
        curr_indices_test_indices = np.array(np.where(w_test_diag_indices))
        d_feature['curr_indices_test_indices_0'] = dtype_feature_value(curr_indices_test_indices[0])  # int64
        d_feature['curr_indices_test_indices_1'] = dtype_feature_value(curr_indices_test_indices[1])  # int64

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()

    return serialized


def np_to_tfrecords(full_b, full_b_test, num_of_samples, L, M, result_tf_file, verbose=True):

    def _dtype_feature(ndarray):
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float32 or dtype_ == np.float64:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        elif dtype_ == np.uint8:
            return lambda array: tf.train.Feature(bytes_list=tf.train.BytesList(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. Instead got {}".format(ndarray.dtype))

    n = full_b.shape[0]

    dtype_feature_value = _dtype_feature(np.array([[1]]))


    # Generate tfrecord writer
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print "Serializing {:d} examples into {}".format(num_of_samples, result_tf_file)

    # iterate over each sample, and serailize it as ProtoBuf.
    for idx in range(num_of_samples):
        d_feature = {}
        curr_node = np.array(random.sample(range(n), 1))
        if random.uniform(0, 1) > 0.5:
            d_feature['edge'] = dtype_feature_value([curr_node, 1])  # int64
            train_edge = sparse.find(full_b[curr_node, :])[1]
            global_mini_batch_indices = train_edge
        else:
            d_feature['edge'] = dtype_feature_value([curr_node, 0])  # int64
            # train_non_edge = sparse.find((full_b[curr_node, :] + full_b_test[curr_node, :]) == 0)[1]
            curr_row = full_b[curr_node, :] + full_b_test[curr_node, :]
            train_non_edge = np.where(curr_row.todense() == 0)[1]
            global_mini_batch_indices = random.sample(list(train_non_edge), len(train_non_edge))[0:L]  # added list to be compatible with 3.7 and 2.7
        ########################## Common ##########################


        d_feature['global_mini_batch_indices'] = dtype_feature_value(global_mini_batch_indices)  # int64
        mini_batch_size = len(global_mini_batch_indices)
        n_rep = np.repeat(n, mini_batch_size + 1)
        M_rep = np.repeat(M, mini_batch_size + 1)
        local_mini_batch_indices_m = np.array(list(map(rand_sample, n_rep, M_rep)))  # use list to be compatible with python 3.7 and 2.7
        d_feature['local_mini_batch_indices_m'] = dtype_feature_value(local_mini_batch_indices_m.reshape(-1))  # int64
        curr_indices = np.append(global_mini_batch_indices, [curr_node])


        row_train_indices = full_b[curr_indices].toarray()
        row_test_indices = full_b_test[curr_indices].toarray()
        row_index = np.array(range(mini_batch_size + 1))
        w_train_indices = row_train_indices[row_index[:, np.newaxis], local_mini_batch_indices_m]
        w_test_diag_indices = row_test_indices[row_index[:, np.newaxis], local_mini_batch_indices_m]


        curr_indices_train_indices = np.array(np.where(w_train_indices))
        if curr_indices_train_indices.size == 0:
            d_feature['curr_indices_train_indices_0'] = dtype_feature_value([])  # int64
            d_feature['curr_indices_train_indices_1'] = dtype_feature_value([])  # int64
            d_feature['true_curr_indices_train_indices_0'] = dtype_feature_value([])  # int64
            d_feature['true_curr_indices_train_indices_1'] = dtype_feature_value([])  # int64
        else:
            d_feature['curr_indices_train_indices_0'] = dtype_feature_value(curr_indices_train_indices[0])  # int64
            d_feature['curr_indices_train_indices_1'] = dtype_feature_value(curr_indices_train_indices[1])  # int64
            true_curr_indices_train_indices_0 = curr_indices[curr_indices_train_indices[0]]
            d_feature['true_curr_indices_train_indices_0'] = dtype_feature_value(true_curr_indices_train_indices_0)  # int64
            true_curr_indices_train_indices_1 = local_mini_batch_indices_m[np.where(w_train_indices)]
            d_feature['true_curr_indices_train_indices_1'] = dtype_feature_value(true_curr_indices_train_indices_1)  # int64
        curr_indices_test_indices = np.array(np.where(w_test_diag_indices))
        d_feature['curr_indices_test_indices_0'] = dtype_feature_value(curr_indices_test_indices[0])  # int64
        d_feature['curr_indices_test_indices_1'] = dtype_feature_value(curr_indices_test_indices[1])  # int64

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print "Writing {} done!".format(result_tf_file)

def read_records(serialized_example):
    feature = {'edge': tf.FixedLenFeature([2, ], tf.int64),
               'global_mini_batch_indices': tf.VarLenFeature(tf.int64),
               'local_mini_batch_indices_m': tf.VarLenFeature(tf.int64),
               'curr_indices_train_indices_0': tf.VarLenFeature(tf.int64),
               'curr_indices_train_indices_1': tf.VarLenFeature(tf.int64),
               'true_curr_indices_train_indices_0': tf.VarLenFeature(tf.int64),
               'true_curr_indices_train_indices_1': tf.VarLenFeature(tf.int64),
               'curr_indices_test_indices_0': tf.VarLenFeature(tf.int64),
               'curr_indices_test_indices_1': tf.VarLenFeature(tf.int64),
               }
    features = tf.parse_single_example(serialized_example, features=feature)
    edge = tf.cast(features['edge'], dtype=tf.int32)
    global_mini_batch_pi = tf.cast(features['global_mini_batch_indices'], dtype=tf.int32)
    local_mini_batch_indices = tf.cast(features['local_mini_batch_indices_m'], dtype=tf.int32)
    curr_indices_train_indices_0 = tf.cast(features['curr_indices_train_indices_0'], dtype=tf.int32)
    curr_indices_train_indices_1 = tf.cast(features['curr_indices_train_indices_1'], dtype=tf.int32)
    true_curr_indices_train_indices_0 = tf.cast(features['true_curr_indices_train_indices_0'], dtype=tf.int32)
    true_curr_indices_train_indices_1 = tf.cast(features['true_curr_indices_train_indices_1'], dtype=tf.int32)
    curr_indices_test_indices_0 = tf.cast(features['curr_indices_test_indices_0'], dtype=tf.int32)
    curr_indices_test_indices_1 = tf.cast(features['curr_indices_test_indices_1'], dtype=tf.int32)
    return edge, global_mini_batch_pi, local_mini_batch_indices, curr_indices_train_indices_0, curr_indices_train_indices_1, true_curr_indices_train_indices_0, true_curr_indices_train_indices_1, curr_indices_test_indices_0, curr_indices_test_indices_1

def assign_record(iterator, global_mini_batch_indices,  m_tf, L_tf,
                 num_train_edge_each_node_tf, local_mini_batch_indices_m, curr_indices_train_indices,
                 true_curr_indices_train_indices, curr_indices_test_indices):
    edge, global_mini_batch_pi, local_mini_batch_indices, curr_indices_train_indices_0, curr_indices_train_indices_1, true_curr_indices_train_indices_0, true_curr_indices_train_indices_1, curr_indices_test_indices_0, curr_indices_test_indices_1 = iterator.get_next()
    curr = edge

    global_mini_batch_indices = tf.assign(global_mini_batch_indices, tf.cast(global_mini_batch_pi.values, tf.int32), validate_shape=False)
    local_mini_batch_indices = tf.cast(local_mini_batch_indices.values, tf.int32)
    num_row_local_mini_batch = tf.cond(tf.equal(curr[0, 1], 1), lambda: num_train_edge_each_node_tf[curr[0, 0]],
                                       lambda: L_tf) + 1
    local_mini_batch_indices_m = tf.assign(local_mini_batch_indices_m,
                                           tf.reshape(local_mini_batch_indices, (num_row_local_mini_batch, m_tf)),
                                           validate_shape=False)

    curr_indices_train_indices = tf.assign(curr_indices_train_indices,
                                           tf.transpose(tf.cast(tf.concat([[curr_indices_train_indices_0.values],
                                                                           [curr_indices_train_indices_1.values]], axis=0),
                                                                tf.int32)), validate_shape=False)
    true_curr_indices_train_indices = tf.assign(true_curr_indices_train_indices,
                                                tf.transpose(tf.cast(tf.concat([[true_curr_indices_train_indices_0.values],
                                                                                [true_curr_indices_train_indices_1.values]], axis=0),
                                                                     tf.int32)), validate_shape=False)
    curr_indices_test_indices = tf.assign(curr_indices_test_indices,
                                          tf.transpose(tf.cast(tf.concat([[curr_indices_test_indices_0.values],
                                                                          [curr_indices_test_indices_1.values]], axis=0), tf.int32)),
                                          validate_shape=False)
    return curr, global_mini_batch_indices, local_mini_batch_indices_m, curr_indices_train_indices, true_curr_indices_train_indices, curr_indices_test_indices
