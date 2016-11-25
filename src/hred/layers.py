import tensorflow as tf
import numpy as np

from initializers import orthogonal_initializer


def embedding_layer(x, name='embedding-layer', vocab_dim=90004, embedding_dim=256):
    """
    Used before the query encoder, to go from the one-hot vocabulary to an embedding
    """

    with tf.variable_scope(name):
        W = tf.get_variable(name="weights", shape=(vocab_dim, embedding_dim),
                            initializer=tf.random_normal_initializer(stddev=0.001))
        embedding = tf.reduce_sum(tf.nn.embedding_lookup(W, x), [1])

    return embedding


def gru_layer_with_reset(h_prev, x, name='gru', x_dim=256, y_dim=512):
    """
    Used for the query encoder layer
    :param x should be a 2-tuple
    """

    # Unpack mandatory packed force_reset_vector
    x, reset_vector = x

    x = tf.reshape(x, [1, x_dim])
    h_prev = tf.reshape(h_prev, [1, y_dim])

    with tf.variable_scope(name):
        h = _gru_layer(h_prev, x, 'gru', x_dim, y_dim)

        # Force reset hidden state
        h_reset = reset_vector * h

    return tf.squeeze(tf.pack([h, h_reset]))


def gru_layer_with_retain(h_prev, x, name='gru', x_dim=256, y_dim=512):
    """
    Used for the session encoder layer
    :param x should be a 2-tuple
    """

    # Unpack mandatory packed retain_vector
    x, retain_vector = x

    h_prev = tf.reshape(h_prev, [1, y_dim])
    x = tf.reshape(x, [1, x_dim])

    with tf.variable_scope(name):
        h = _gru_layer(h_prev, x, 'gru', x_dim, y_dim)

        # Force reset hidden state
        h_retain = retain_vector * h_prev + (1 - retain_vector) * h

    return tf.squeeze(tf.pack([h, h_retain]))


def gru_layer_with_state_reset(h_prev, x, name='gru', x_dim=256, h_dim=512, y_dim=1024):
    """
    Used for the decoder layer
    :param x should be a 3-tuple
    """

    # Unpack mandatory packed retain_vector and the state
    x, retain_vector, state = x

    h_prev = tf.reshape(h_prev, [1, y_dim])
    state = tf.reshape(state, [1, h_dim])
    x = tf.reshape(x, [1, x_dim])

    with tf.variable_scope(name):

        with tf.variable_scope('state_start'):
            W = tf.get_variable(name='weight', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))

            h_prev_state = retain_vector * h_prev + (1 - retain_vector) * tf.tanh(tf.matmul(state, W) + b)

        h = _gru_layer_with_state(h_prev_state, x, state, 'gru', x_dim, y_dim, h_dim)

    return tf.squeeze(h)


def output_layer(h, x, name='output', x_dim=256, y_dim=512, h_dim=512):
    """
    Used after the decoder
    """

    with tf.variable_scope(name):
        Wh = tf.get_variable(name='weight_hidden', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
        Wi = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
        b = tf.get_variable(name='bias_input', shape=(y_dim,), initializer=tf.random_normal_initializer(stddev=0.001))

        y = tf.matmul(h, Wh) + tf.matmul(x, Wi) + b

    return y


def softmax_layer(x, name='softmax', x_dim=512, y_dim=90004):

    with tf.variable_scope(name):

        W = tf.get_variable(name='weight', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
        b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.random_normal_initializer(stddev=0.001))

        y = tf.nn.softmax(tf.matmul(x, W) + b)

    return y


def _gru_layer(h_prev, x, name='gru', x_dim=256, y_dim=512):
    """
    Used for both encoder layers
    """

    with tf.variable_scope(name):

        # Reset gate
        with tf.variable_scope('reset_gate'):

            Wi_r = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            Wh_r = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=orthogonal_initializer())
            b_r = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            r = tf.sigmoid(tf.matmul(x, Wi_r)) + tf.matmul(h_prev, Wh_r) + b_r

        # Update gate
        with tf.variable_scope('update_gate'):
            Wi_z = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            Wh_z = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=orthogonal_initializer())
            b_z = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            z = tf.sigmoid(tf.matmul(x, Wi_z)) + tf.matmul(h_prev, Wh_z) + b_z

        # Candidate update
        with tf.variable_scope('candidate_update'):
            Wi_h_tilde = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            Wh_h_tilde = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=orthogonal_initializer())
            b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            h_tilde = tf.tanh(tf.matmul(x, Wi_h_tilde)) + tf.matmul(r * h_prev, Wh_h_tilde) + b_h_tilde

        # Final update
        h = (np.float32(1.0) - z) * h_prev + z * h_tilde

    return h


def _gru_layer_with_state(h_prev, x, state, name='gru', x_dim=256, y_dim=1024, h_dim=512):
    """
    Used for decoder
    """

    with tf.variable_scope(name):

        # Reset gate
        with tf.variable_scope('reset_gate'):
            Wi_r = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            Wh_r = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=orthogonal_initializer())
            Ws_r = tf.get_variable(name='weight_state', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            b_r = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            r = tf.sigmoid(tf.matmul(x, Wi_r)) + tf.matmul(h_prev, Wh_r) + tf.matmul(state, Ws_r) + b_r

        # Update gate
        with tf.variable_scope('update_gate'):
            Wi_z = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            Wh_z = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=orthogonal_initializer())
            Ws_r = tf.get_variable(name='weight_state', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            b_z = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            z = tf.sigmoid(tf.matmul(x, Wi_z)) + tf.matmul(h_prev, Wh_z) + tf.matmul(state, Ws_r) + b_z

        # Candidate update
        with tf.variable_scope('candidate_update'):
            Wi_h_tilde = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            Wh_h_tilde = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=orthogonal_initializer())
            Ws_h_tilde = tf.get_variable(name='weight_state', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.001))
            b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            h_tilde = tf.tanh(tf.matmul(x, Wi_h_tilde)) + \
                      tf.matmul(r * h_prev, Wh_h_tilde) + \
                      tf.matmul(state, Ws_h_tilde) + \
                      b_h_tilde

        # Final update
        h = (np.float32(1.0) - z) * h_prev + z * h_tilde

    return h