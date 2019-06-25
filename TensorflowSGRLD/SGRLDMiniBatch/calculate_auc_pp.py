import tensorflow as tf



def get_pp(log_sum_probs, true_y):
    num_test_links = tf.shape(true_y)[0]
    non_edge = tf.gather(log_sum_probs, tf.where(tf.equal(true_y, False)))
    edge = tf.gather(log_sum_probs, tf.where(tf.equal(true_y, True)))
    sum_test_ll = tf.reduce_sum(non_edge, 0) + tf.reduce_sum(tf.log(1-tf.exp(edge)), 0)
    pp = tf.exp(tf.divide(-sum_test_ll, tf.cast(num_test_links, tf.float32)))
    return pp



def get_roc_auc(num_probs, roc_y):
    count_nonzero = tf.count_nonzero(roc_y)
    count_zero = tf.count_nonzero(tf.logical_not(roc_y))
    stack_x = tf.divide(tf.cumsum(tf.cast(tf.equal(roc_y, False), tf.float32)), tf.cast(count_zero, tf.float32))
    stack_y = tf.divide(tf.cumsum(tf.cast(tf.equal(roc_y, True), tf.float32)), tf.cast(count_nonzero, tf.float32))
    a = tf.slice(stack_x, begin=[1], size=[num_probs-1]) - tf.slice(stack_x, begin=[0], size=[num_probs-1])
    b = tf.slice(stack_y, begin=[1], size=[num_probs-1])
    c = tf.reduce_sum(tf.multiply(a, b))
    return c

def calculate_auc_pp(w_tf, test_pair_tf, test_links_tf, pi_0_tf, pi_tf):
    indices_1 = tf.gather(test_pair_tf, indices=0, axis=0)
    indices_2 = tf.gather(test_pair_tf, indices=1, axis=0)
    start_node_prob = tf.gather(w_tf, indices=indices_1, axis=0)
    end_node_prob = tf.gather(w_tf, indices=indices_2, axis=0)
    log_sum_test = tf.log(1 - pi_0_tf) + tf.reduce_sum(
        tf.log(1 - tf.multiply(tf.multiply(start_node_prob, end_node_prob), pi_tf)), axis=1)
    # PP
    pp = get_pp(log_sum_test, test_links_tf)
    # AUC
    curr_test_pair_prob = 1 - tf.exp(log_sum_test)
    val, idx = tf.nn.top_k(curr_test_pair_prob, k=tf.shape(curr_test_pair_prob)[0])
    roc_y = tf.gather(test_links_tf, indices=idx)
    num_probs = tf.shape(curr_test_pair_prob)[0]
    auc_roc_calculated = get_roc_auc(num_probs, roc_y)
    # return [auc_roc_calculated, pp, roc_y[0:10], val[0:10], roc_y[num_probs-10:num_probs-1], val[num_probs-10:num_probs-1]]
    return [auc_roc_calculated, pp]

def calculate_auc_pp_collections(w_tf, test_pair_tf, test_links_tf, pi_0_tf, pi_tf, total_test_edge_prob_tf):
    indices_1 = tf.gather(test_pair_tf, indices=0, axis=0)
    indices_2 = tf.gather(test_pair_tf, indices=1, axis=0)
    start_node_prob = tf.gather(w_tf, indices=indices_1, axis=0)
    end_node_prob = tf.gather(w_tf, indices=indices_2, axis=0)
    log_sum_test = tf.log(1 - pi_0_tf) + tf.reduce_sum(
        tf.log(1 - tf.multiply(tf.multiply(start_node_prob, end_node_prob), pi_tf)), axis=1)
    # PP
    pp = get_pp(log_sum_test, test_links_tf)
    # AUC
    curr_test_pair_prob = 1 - tf.exp(log_sum_test)
    assign_total_test_edge_prob_tf = total_test_edge_prob_tf.assign(total_test_edge_prob_tf + curr_test_pair_prob)
    val, idx = tf.nn.top_k(curr_test_pair_prob, k=tf.shape(curr_test_pair_prob)[0])
    roc_y = tf.gather(test_links_tf, indices=idx)
    num_probs = tf.shape(curr_test_pair_prob)[0]
    auc_roc_calculated = get_roc_auc(num_probs, roc_y)
    return [auc_roc_calculated, pp, assign_total_test_edge_prob_tf]

def calculate_avg_auc(total_test_edge_prob_tf, test_links_tf, count):
    # AUC
    curr_test_pair_prob = tf.divide(total_test_edge_prob_tf, tf.cast(count, dtype=tf.float32))
    val, idx = tf.nn.top_k(curr_test_pair_prob, k=tf.shape(curr_test_pair_prob)[0])
    roc_y = tf.gather(test_links_tf, indices=idx)
    num_probs = tf.shape(curr_test_pair_prob)[0]
    auc_roc_calculated = get_roc_auc(num_probs, roc_y)
    return [auc_roc_calculated]