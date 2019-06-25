
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import numpy as np
import timeit
import tensorflow as tf
from scipy import sparse
import os
from argparse import ArgumentParser

from sample_initial_parameters import sample_initial_parameters_pi_alpha, sample_initial_parameters_w

from util import set_seed
from create_mask_network_sparse_equal import create_mask_network_sparse_equal
from sample_alpha_gibbs_sagm import sample_alpha_gibbs_sagm
from calculate_auc_pp import calculate_auc_pp, calculate_auc_pp_collections, calculate_avg_auc

# for tensorflow records
from tf_records import np_to_tfrecords, read_records, assign_record

# for mini-batch
from compute_mini_gradient_pi import compute_mini_gradient_pi_edge, compute_mini_gradient_pi_non_edge
from compute_mini_gradient_w import compute_mini_gradient_w


def main(_):
    # Input Arguments
    parser = ArgumentParser()
    parser.add_argument('-seed', dest="seed", type=int, help='Seed of random generator')
    parser.add_argument('-n', '--network', dest="network", help='The name of the network')
    parser.add_argument("-o", "--outfile", dest="outfile", help="The name of the output file")
    parser.add_argument('-k', dest="K", type=int, help="Number of communities")
    parser.add_argument('-l', dest="L", type=int, help="Mini-batch size")
    parser.add_argument('-m', dest="m", type=int, help="Mini-batch per node")
    parser.add_argument('-ns', dest="num_of_samples_in_records", type=int, help="Number of samples in a records")
    parser.add_argument('-b', '--burn_in', dest="num_burn_in", type=int, help="Number of burn in")
    parser.add_argument('-s', '--num_samples', dest="num_samples", type=int,
                        help="Number of samples collected after burn in")
    parser.add_argument('-tr', '--test_ratio', dest="Test_ratio", type=float, help="Number of Test node pairs to compute AUC")
    parser.add_argument("-q", "--quiet",
                        action="store_false", dest="verbose", default=True,
                        help="don't print status messages to stdout")


    args = parser.parse_args()
    print(args)
    network = args.network
    outfile = args.outfile
    k = args.K
    L = args.L
    M = args.m
    num_of_samples_in_records = args.num_of_samples_in_records
    num_burn_in = args.num_burn_in
    num_samples = args.num_samples
    Test_ratio = args.Test_ratio
    seed = args.seed

    outfile = outfile + network + 'Seed' + str(seed) + 'L' + str(L) + 'M' + str(M) + 'K' + str(k) + 'burn_in' + str(
        num_burn_in) + 'samples' + str(num_samples) + 'test_ratio' + str(Test_ratio) + '.npz'
    print(outfile)

    # Parameters of the model
    # Set up the parameters for training
    eta_0 = 5
    eta_1 = 1

    step_size_a_w = 0.001
    step_size_a_pi = 0.001
    step_size_b = 1000
    step_size_c = 0.55

    # Create the necessary directory
    if os.path.isdir('../Results') is False:
        os.mkdir('../Results')
    if os.path.isdir('../records') is False:
        os.mkdir('../records')

    network_file_name = '../Datasets/' + network + '.npz'
    B = sparse.load_npz(network_file_name)


    n = B.shape[1]
    Num_edges = B.count_nonzero()
    print('Number of nodes is {} and number of edges is {}'.format(n, Num_edges))
    Test = Test_ratio*Num_edges
    Train = Test

    num_tensorflow_records = np.ceil((num_burn_in + num_samples) / num_of_samples_in_records)

    # set seed for both np and tensorflow
    set_seed(seed)

    alpha = np.float32(np.ones((1, k)))
    w, sum_w_bar = sample_initial_parameters_w(k, n)


    # Find the test set
    [test_pairs, train_pairs] = create_mask_network_sparse_equal(B, Test, Train)
    total_test_edge_prob = np.zeros(test_pairs.shape[1])
    Boolean_test_links = np.squeeze(np.array(B[test_pairs[0, :], test_pairs[1, :]]))
    Boolean_train_links = np.squeeze(np.array(B[train_pairs[0, :], train_pairs[1, :]]))

    B[test_pairs[0, :], test_pairs[1, :]] = 1
    full_b = np.transpose(B) + B
    num_train_non_edge_each_node = (n-1)-np.squeeze(np.array(np.sum(full_b, axis=1)))  # 1 is for self-loop
    min_num_train_non_edge_each_node = min(num_train_non_edge_each_node)
    print("Minimum number of non edges per node : {}".format(min_num_train_non_edge_each_node))
    B[test_pairs[0, :], test_pairs[1, :]] = 0
    full_b = np.transpose(B) + B
    num_train_edge_each_node = np.squeeze(np.array(np.sum(full_b, axis=1)))
    print("Maximum number of edges per node : {}".format(max(num_train_edge_each_node)))


    B = sparse.csr_matrix((np.repeat(True, test_pairs.shape[1]), (test_pairs[0, :], test_pairs[1, :])), shape=(n, n))
    full_b_test = np.transpose(B) + B
    full_b_test.setdiag(True)

    # Assert L is less than train non-edge for every node
    if L > min_num_train_non_edge_each_node:
        print("L should be less than train non-edge for every node")
        exit()

    # Tensorflow variables
    pi_tf, sum_pi_bar_tf, alpha_tf = sample_initial_parameters_pi_alpha(k, alpha, eta_0, eta_1)
    w_holder = tf.placeholder(tf.float32)
    sum_w_holder = tf.placeholder(tf.float32)
    w_tf = tf.Variable(w_holder, validate_shape=False, dtype=tf.float32)
    sum_w_bar_tf = tf.Variable(sum_w_holder, validate_shape=False, dtype=tf.float32)

    test_pairs_tf = tf.convert_to_tensor(test_pairs, dtype=tf.int32, name="test_pairs_tf")
    boolean_test_links_tf = tf.convert_to_tensor(Boolean_test_links, dtype=tf.bool, name="boolean_test_links_tf")
    train_pairs_tf = tf.convert_to_tensor(train_pairs, dtype=tf.int32, name="test_pairs_tf")
    boolean_train_links_tf = tf.convert_to_tensor(Boolean_train_links, dtype=tf.bool, name="boolean_test_links_tf")
    beta_0_tf = tf.constant(1., dtype=tf.float32, name="alpha_shape_tf")
    beta_1_tf = tf.constant(1., dtype=tf.float32, name="alpha_rate_tf")
    n_tf = tf.constant(n, dtype=tf.float32, name="n_tf")
    k_tf = tf.constant(k, dtype=tf.int32, name="k_tf")
    eta_0_by_1_tf = tf.constant([[eta_0], [eta_1]], dtype=tf.float32, name="eta_0_tf")
    pi_0_tf = tf.constant(0.00005, dtype=tf.float32, name="pi_0_tf")

    step_size_w_tf = tf.placeholder(dtype=tf.float32, name="step_size_w_tf")
    step_size_pi_tf = tf.placeholder(dtype=tf.float32, name="step_size_pi_tf")

    # Tensorflow variables specific for mini-batch
    global_mini_batch_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="global_mini_batch_indices")
    local_mini_batch_indices_m = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="local_mini_batch_indices_m")
    curr_indices_train_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="curr_indices_train_indices")
    true_curr_indices_train_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="true_curr_indices_train_indices")
    curr_indices_test_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="curr_indices_test_indices")
    total_test_edge_prob_tf = tf.Variable(total_test_edge_prob, total_test_edge_prob.shape, dtype=tf.float32, name="total_test_edge_prob_tf")
    file_name_tf = tf.Variable('abc', dtype=tf.string)

    L_tf = tf.constant(L, dtype=tf.int32, name="L_tf")
    m_tf = tf.constant(M, dtype=tf.int32, name="m_tf")
    num_train_edge_each_node_tf = tf.constant(num_train_edge_each_node, dtype=tf.int32, name="m_tf")

    curr_node_tf = tf.placeholder(dtype=tf.int32, name="curr_node_tf")
    mini_batch_size_tf = tf.placeholder(dtype=tf.int32, name="mini_batch_size_tf")
    count_tf = tf.placeholder(dtype=tf.int32, name="count_tf")


    # Tensorflow Computational Graph
    auc_pp_test = calculate_auc_pp(w_tf, test_pairs_tf, boolean_test_links_tf, pi_0_tf, pi_tf)
    auc_pp_test_collections = calculate_auc_pp_collections(w_tf, test_pairs_tf, boolean_test_links_tf, pi_0_tf, pi_tf, total_test_edge_prob_tf)
    avg_auc_test = calculate_avg_auc(total_test_edge_prob_tf, boolean_test_links_tf, count_tf)
    auc_pp_train = calculate_auc_pp(w_tf, train_pairs_tf, boolean_train_links_tf, pi_0_tf, pi_tf)

    find_mini_gradient_pi_edge = compute_mini_gradient_pi_edge(pi_tf, sum_pi_bar_tf, pi_0_tf, w_tf, curr_node_tf, n_tf,
                                                               k_tf, step_size_pi_tf, eta_0_by_1_tf, global_mini_batch_indices)

    find_mini_gradient_pi_non_edge = compute_mini_gradient_pi_non_edge(pi_tf, sum_pi_bar_tf, L_tf, w_tf, curr_node_tf,
                                                                       n_tf, k_tf, step_size_pi_tf, eta_0_by_1_tf,
                                                                       global_mini_batch_indices)


    find_mini_gradient_w = compute_mini_gradient_w(w_tf, sum_w_bar_tf, pi_tf, pi_0_tf, alpha_tf, curr_node_tf, n_tf,
                                                   k_tf, step_size_w_tf, mini_batch_size_tf, m_tf,
                                                   local_mini_batch_indices_m, global_mini_batch_indices,
                                                   curr_indices_train_indices, true_curr_indices_train_indices,
                                                   curr_indices_test_indices)

    find_alpha = sample_alpha_gibbs_sagm(alpha_tf, beta_0_tf, beta_1_tf, w_tf, n_tf)

    dataset = tf.data.TFRecordDataset([file_name_tf])
    dataset = dataset.map(read_records)
    dataset = dataset.batch(1)
    dataset = dataset.shuffle(buffer_size=50)
    dataset = dataset.repeat(1)
    iterator = dataset.make_initializable_iterator()
    edge = assign_record(iterator, global_mini_batch_indices,  m_tf, L_tf, num_train_edge_each_node_tf,
                                        local_mini_batch_indices_m, curr_indices_train_indices,
                                        true_curr_indices_train_indices, curr_indices_test_indices)



    # Output vectors to be saved
    TestAUCvector = []
    TrainAUCvector = []
    TestPPvector = []
    TrainPPvector = []
    TimeVector = []

    count_sample_auc = 0

    start = timeit.default_timer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create the tensorflow records
        record_file = '../records/' + network + '.tfrecords'
        np_to_tfrecords(full_b, full_b_test, num_of_samples_in_records, L, M, record_file, verbose=False)

        # Initialize the variables of the model
        init = tf.global_variables_initializer()
        sess.run(init, feed_dict={w_holder: w, sum_w_holder: sum_w_bar})

        # Initialize the record
        record_number = 0
        sess.run(iterator.initializer, feed_dict={file_name_tf: record_file})
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("Starting the burn_in period")
        for iteration in range(num_burn_in):
            # print("Current iteration {}".format(iteration))
            step_size_w = step_size_a_w * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            step_size_pi = step_size_a_pi * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            # MINI-BATCH SAMPLING
            curr, _, _, _, _, _ = sess.run(edge)
            curr_node = curr[0, 0]
            sample_edge = curr[0, 1] == 1
            # print(curr_node)
            # print(sample_edge)
            if sample_edge:
                # Sample w and pi if train edges of curr_node
                sess.run(find_mini_gradient_w, feed_dict={curr_node_tf: curr_node, step_size_w_tf: step_size_w,
                                                          mini_batch_size_tf: num_train_edge_each_node[curr_node]})
                pi,_ = sess.run(find_mini_gradient_pi_edge, feed_dict={curr_node_tf: curr_node, step_size_pi_tf: step_size_pi})

            else:
                # Sample w and pi if train non-edges of curr_node
                sess.run(find_mini_gradient_w, feed_dict={curr_node_tf: curr_node, step_size_w_tf: step_size_w,
                                                          mini_batch_size_tf: L})
                pi,_ = sess.run(find_mini_gradient_pi_non_edge,
                         feed_dict={curr_node_tf: curr_node, step_size_pi_tf: step_size_pi})

            # SAMPLING ALPHA
            sess.run(find_alpha)
            # COMPUTING AUC
            if (iteration+1) % num_of_samples_in_records == 0:
                print("The current iteration is " + str(iteration+1))
                result_test = sess.run(auc_pp_test)
                TestAUCvector.append(result_test[0])
                TestPPvector.append(result_test[1])
                print("Printing results for Test set")
                print("auc tf : " + str(result_test[0]))
                print("pp tf : " + str(result_test[1]))
                result_train = sess.run(auc_pp_train)
                TrainAUCvector.append(result_train[0])
                TrainPPvector.append(result_train[1])
                print("Printing results for Train set")
                print("auc tf : " + str(result_train[0]))
                print("pp tf : " + str(result_train[1]))
                record_number = record_number + 1
                stop_iter = timeit.default_timer()
                TimeVector.append(stop_iter-start)
                if num_tensorflow_records > record_number:
                    # create the record
                    record_file = '../records/' + network + '.tfrecords'
                    np_to_tfrecords(full_b, full_b_test, num_of_samples_in_records, L, M, record_file, verbose=False)
                    # initilize the record
                    sess.run(iterator.initializer, feed_dict={file_name_tf: record_file})


        print("Starting collecting samples")
        for iteration in range(num_burn_in, num_burn_in + num_samples):
            # print("Current iteration {}".format(iteration))
            step_size_w = step_size_a_w * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            step_size_pi = step_size_a_pi * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            # MINI-BATCH SAMPLING
            curr, _, _, _, _, _ = sess.run(edge)
            curr_node = curr[0, 0]
            sample_edge = curr[0, 1] == 1
            # print(curr_node)
            # print(sample_edge)
            if sample_edge:
                # Sample w and pi if train edges of curr_node
                sess.run(find_mini_gradient_w, feed_dict={curr_node_tf: curr_node, step_size_w_tf: step_size_w,
                                                          mini_batch_size_tf: num_train_edge_each_node[curr_node]})
                pi,_ = sess.run(find_mini_gradient_pi_edge, feed_dict={curr_node_tf: curr_node, step_size_pi_tf: step_size_pi})

            else:
                # Sample w and pi if train non-edges of curr_node
                sess.run(find_mini_gradient_w, feed_dict={curr_node_tf: curr_node, step_size_w_tf: step_size_w,
                                                          mini_batch_size_tf: L})
                pi,_ = sess.run(find_mini_gradient_pi_non_edge,
                         feed_dict={curr_node_tf: curr_node, step_size_pi_tf: step_size_pi})

            # SAMPLING ALPHA
            sess.run(find_alpha)
            # COMPUTING AUC
            if (iteration + 1) % num_of_samples_in_records == 0:
                count_sample_auc = count_sample_auc + 1
                print("The current iteration is " + str(iteration + 1))
                result_test = sess.run(auc_pp_test_collections)
                TestAUCvector.append(result_test[0])
                TestPPvector.append(result_test[1])
                print("Printing results for Test set")
                print("auc tf : " + str(result_test[0]))
                print("pp tf : " + str(result_test[1]))
                result_train = sess.run(auc_pp_train)
                TrainAUCvector.append(result_train[0])
                TrainPPvector.append(result_train[1])
                print("Printing results for Train set")
                print("auc tf : " + str(result_train[0]))
                print("pp tf : " + str(result_train[1]))
                record_number = record_number + 1
                stop_iter = timeit.default_timer()
                TimeVector.append(stop_iter - start)
                print(stop_iter)
                if num_tensorflow_records > record_number:
                    # create the record
                    record_file = '../records/' + network + '.tfrecords'
                    np_to_tfrecords(full_b, full_b_test, num_of_samples_in_records, L, M, record_file, verbose=False)
                    # initilize the record
                    sess.run(iterator.initializer, feed_dict={file_name_tf: record_file})

        avg_auc_test = sess.run(avg_auc_test, feed_dict={count_tf: count_sample_auc})
        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print('Avg auc : ', avg_auc_test)
    print(TestAUCvector)
    np.savez(outfile, AvgAUC=avg_auc_test,TestAUCvector=TestAUCvector,TestPPvector=TestPPvector,TrainAUCvector=TrainAUCvector,TrainPPvector=TrainPPvector,TimeVector=TimeVector)

if __name__ == "__main__":
    tf.app.run()
