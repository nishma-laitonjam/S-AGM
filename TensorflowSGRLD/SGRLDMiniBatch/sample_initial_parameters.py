import numpy as np
import tensorflow as tf

def sample_initial_parameters_pi_alpha(k, alpha, eta_0, eta_1):
    pi_bar_gamma_shape = np.float32(np.repeat([np.array([eta_0, eta_1])], [k], axis=0).T)  # 2*K matrix
    pi_bar = np.float32(np.random.gamma(pi_bar_gamma_shape, 1))

    sum_pi_bar = np.sum(pi_bar, axis=0).reshape((1, k))
    pi = np.divide(pi_bar[0, :], sum_pi_bar)

    pi_tf = tf.Variable(pi, dtype=tf.float32, name="pi_tf")
    sum_pi_bar_tf = tf.Variable(sum_pi_bar, dtype=tf.float32, name="sum_pi_bar_tf")

    alpha_tf = tf.Variable(alpha, dtype=tf.float32, name="alpha_tf")


    return pi_tf, sum_pi_bar_tf, alpha_tf

def sample_initial_parameters_w(k, n):
    sum_w_bar = np.ones((n, k))
    w = np.divide(np.ones((n, k)), k)

    return w, sum_w_bar