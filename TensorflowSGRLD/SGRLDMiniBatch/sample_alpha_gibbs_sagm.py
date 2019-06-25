import tensorflow as tf

def sample_alpha_gibbs_sagm(alpha_tf, beta_0_tf, beta_1_tf, w_tf, n_tf):
    rate = beta_1_tf - tf.reduce_sum(tf.log(w_tf), axis=0)
    alpha = tf.random_gamma(shape=(1, ), alpha=n_tf + beta_0_tf, beta=rate)
    assign_alpha_tf = alpha_tf.assign(alpha)
    return assign_alpha_tf