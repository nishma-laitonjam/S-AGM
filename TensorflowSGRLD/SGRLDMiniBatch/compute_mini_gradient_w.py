import tensorflow as tf

def compute_mini_gradient_w(w_tf, sum_w_bar_tf, pi_tf, pi_0_tf, alpha_tf, curr_node_tf, n_tf, k_tf, step_size_w_tf,
                            mini_batch_size_tf, m_tf, local_mini_batch_indices_m, global_mini_batch_indices,
                            curr_indices_train_indices, true_curr_indices_train_indices, curr_indices_test_indices):

    w_train_reduced = tf.sparse_to_dense(sparse_indices=curr_indices_train_indices, sparse_values=2.,
                                         output_shape=[mini_batch_size_tf + 1, m_tf])
    w_test_diag_reduced = tf.sparse_to_dense(sparse_indices=curr_indices_test_indices, sparse_values=1.,
                                             output_shape=[mini_batch_size_tf + 1, m_tf])
    curr_w_mask = 1 - (w_train_reduced + w_test_diag_reduced)

    wwpi_train_edge = tf.multiply(tf.gather(w_tf, indices=true_curr_indices_train_indices[:, 0], axis=0),
                                  tf.multiply(tf.gather(w_tf, indices=true_curr_indices_train_indices[:, 1], axis=0),
                                              pi_tf))
    curr_log_train_edge = tf.reduce_sum(tf.log(1 - wwpi_train_edge), axis=1) + tf.log(1 - pi_0_tf)

    curr_b_reduced_m = tf.sparse_to_dense(sparse_indices=curr_indices_train_indices,
                                          sparse_values=curr_log_train_edge - tf.log(1 - tf.exp(curr_log_train_edge)),
                                          output_shape=[mini_batch_size_tf + 1, m_tf])

    h = tf.multiply(curr_w_mask, tf.exp(curr_b_reduced_m))

    # Find train edges with curr node
    curr_indices = tf.concat([global_mini_batch_indices, [curr_node_tf]], axis=0)

    # Current w and sum_w_bar to be updated
    curr_w_tf = tf.gather(w_tf, indices=curr_indices, axis=0)
    curr_sum_w_bar_tf = tf.gather(sum_w_bar_tf, indices=curr_indices, axis=0)
    w_pi_reduced_m = tf.transpose(tf.multiply(tf.gather(w_tf, local_mini_batch_indices_m, axis=0), pi_tf), perm=[0, 2, 1])
    ww_pi_reduced_m = tf.multiply(tf.expand_dims(curr_w_tf, -1), w_pi_reduced_m)

    # Common Term
    common_term_reduced_m = tf.divide(-w_pi_reduced_m, 1 - ww_pi_reduced_m)
    row_sum_reduced_m = tf.reduce_sum(tf.multiply(common_term_reduced_m, tf.expand_dims(h, 1)), axis=2)

    updated_w = tf.divide((1 - curr_w_tf), curr_sum_w_bar_tf)
    updated_sum_w_bar = tf.divide((- curr_w_tf), curr_sum_w_bar_tf)

    # Gradient
    scaled_gradient_w_bar0 = tf.multiply(tf.divide(n_tf, tf.cast(m_tf, dtype=tf.float32)),
                                         tf.multiply(updated_w, row_sum_reduced_m))
    scaled_gradient_w_bar1 = tf.multiply(tf.divide(n_tf, tf.cast(m_tf, dtype=tf.float32)),
                                         tf.multiply(updated_sum_w_bar, row_sum_reduced_m))

    #  Gaussian noise
    xi_0 = tf.random_normal(shape=(mini_batch_size_tf + 1, k_tf), mean=0.,
                            stddev=tf.sqrt(step_size_w_tf), dtype=tf.float32, name="xi_0")
    xi_1 = tf.random_normal(shape=(mini_batch_size_tf + 1, k_tf), mean=0.,
                            stddev=tf.sqrt(step_size_w_tf), dtype=tf.float32, name="xi_1")

    # Sampling
    curr_w_bar_0 = tf.multiply(curr_w_tf, curr_sum_w_bar_tf)
    curr_w_bar_1 = tf.multiply((1 - curr_w_tf), curr_sum_w_bar_tf)

    # New samples
    curr_w_bar_0_new = tf.abs(curr_w_bar_0 + tf.multiply(tf.sqrt(curr_w_bar_0), xi_0) +
                              tf.multiply(alpha_tf - curr_w_bar_0 + scaled_gradient_w_bar0, tf.divide(step_size_w_tf, 2)))

    curr_w_bar_1_new = tf.abs(curr_w_bar_1 + tf.multiply(tf.sqrt(curr_w_bar_1), xi_1) +
                              tf.multiply(1 - curr_w_bar_1 + scaled_gradient_w_bar1, tf.divide(step_size_w_tf, 2)))

    curr_w_bar_sum_new = curr_w_bar_0_new + curr_w_bar_1_new
    curr_w_new = tf.divide(curr_w_bar_0_new, curr_w_bar_sum_new)

    w_tf = tf.scatter_update(w_tf, indices=curr_indices, updates=curr_w_new)
    sum_w_bar_tf = tf.scatter_update(sum_w_bar_tf, indices=curr_indices, updates=curr_w_bar_sum_new)

    return [w_tf, sum_w_bar_tf]

