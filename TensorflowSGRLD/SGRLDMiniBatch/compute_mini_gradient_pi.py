import tensorflow as tf

def compute_mini_gradient_pi_edge(pi_tf, sum_pi_bar_tf, pi_0_tf, w_tf, curr_node_tf, n_tf, k_tf, step_size_pi_tf,
                                  eta_0_by_1_tf, global_mini_batch_indices):
    #  Calculating H
    ww_train_edge = tf.multiply(tf.gather(w_tf, indices=curr_node_tf, axis=0),
                                tf.gather(w_tf, indices=global_mini_batch_indices, axis=0))
    log_term_train_edge = tf.log(1 - tf.multiply(ww_train_edge, pi_tf))
    sum_log_term_train_edge = tf.reduce_sum(log_term_train_edge, axis=1, keepdims=True) + tf.log(1 - pi_0_tf)
    h = tf.divide(tf.multiply(ww_train_edge, tf.exp(sum_log_term_train_edge - log_term_train_edge)),
                  (1 - tf.exp(sum_log_term_train_edge)))

    #  Calculating G
    common_term = tf.divide(tf.concat((1 - pi_tf, -pi_tf), axis=0), sum_pi_bar_tf)  # 2 * K array
    gradient_pi_bar_tf = tf.multiply(common_term, tf.reduce_sum(h, axis=0))  # 2 * K array

    scaled_gradient_pi_bar_tf = tf.multiply(n_tf, gradient_pi_bar_tf)

    pi_bar = tf.multiply(tf.concat((pi_tf, 1 - pi_tf), axis=0), sum_pi_bar_tf)  # 2 * K array # change
    xi = tf.random_normal(shape=(2, k_tf), mean=0., stddev=tf.sqrt(step_size_pi_tf),
                          dtype=tf.float32, name="xi")
    updated_pi_bar = tf.abs(pi_bar + tf.multiply(tf.sqrt(pi_bar), xi) +
                            tf.multiply(eta_0_by_1_tf - pi_bar + scaled_gradient_pi_bar_tf, tf.divide(step_size_pi_tf, 2)))

    updated_sum_pi_bar = tf.reduce_sum(updated_pi_bar, axis=0, keepdims=True)  # 1 * K array
    updated_pi = tf.divide(updated_pi_bar[0, :], updated_sum_pi_bar)  # 1* K array

    assign_pi_tf = pi_tf.assign(updated_pi)
    assign_sum_pi_bar_tf = sum_pi_bar_tf.assign(updated_sum_pi_bar)
    return [assign_pi_tf, assign_sum_pi_bar_tf]


def compute_mini_gradient_pi_non_edge(pi_tf, sum_pi_bar_tf, L_tf, w_tf, curr_node_tf, n_tf, k_tf, step_size_pi_tf,
                                      eta_0_by_1_tf, global_mini_batch_indices):

    #  Calculating H
    ww_train_non_edge = tf.multiply(tf.gather(w_tf, indices=curr_node_tf, axis=0),
                                    tf.gather(w_tf, indices=global_mini_batch_indices, axis=0))
    h = tf.divide(-ww_train_non_edge, 1 - tf.multiply(ww_train_non_edge, pi_tf))

    #  Calculating G
    common_term = tf.divide(tf.concat((1 - pi_tf, -pi_tf), axis=0), sum_pi_bar_tf)  # 2 * K array
    gradient_pi_bar_tf = tf.multiply(common_term, tf.reduce_sum(h, axis=0))  # 2 * K array

    scaled_gradient_pi_bar_tf = tf.multiply(tf.divide(tf.multiply(n_tf, n_tf), tf.cast(L_tf, dtype=tf.float32)),
                                            gradient_pi_bar_tf)  # 2 * K array

    pi_bar = tf.multiply(tf.concat((pi_tf, 1 - pi_tf), axis=0), sum_pi_bar_tf)  # 2 * K array
    xi = tf.random_normal(shape=(2, k_tf), mean=0., stddev=tf.sqrt(step_size_pi_tf),
                          dtype=tf.float32, name="xi")
    updated_pi_bar = tf.abs(pi_bar + tf.multiply(tf.sqrt(pi_bar), xi) +
                            tf.multiply(eta_0_by_1_tf - pi_bar + scaled_gradient_pi_bar_tf, tf.divide(step_size_pi_tf, 2)))

    updated_sum_pi_bar = tf.reduce_sum(updated_pi_bar, axis=0, keepdims=True)  # 1 * K array
    updated_pi = tf.divide(updated_pi_bar[0, :], updated_sum_pi_bar) # 1* K array

    assign_pi_tf = pi_tf.assign(updated_pi)
    assign_sum_pi_bar_tf = sum_pi_bar_tf.assign(updated_sum_pi_bar)
    return [assign_pi_tf, assign_sum_pi_bar_tf]
