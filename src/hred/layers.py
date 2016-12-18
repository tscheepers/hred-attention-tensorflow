import tensorflow as tf
import numpy as np

import initializer


def embedding_layer(x, name='embedding-layer', vocab_dim=90004, embedding_dim=256, reuse=None):
    """
    Used before the query encoder, to go from the vocabulary to an embedding
    """

    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(name="weights", shape=(vocab_dim, embedding_dim),
                            initializer=tf.random_normal_initializer(stddev=0.01))
        embedding = tf.nn.embedding_lookup(W, x)

    return embedding


def gru_layer_with_reset(h_prev, x_packed, name='gru', x_dim=256, y_dim=512, reuse=None):
    """
    Used for the query encoder layer. The encoder is reset after an EoQ symbol
    has been reached.

    :param h_prev: previous state of the GRU layer
    :param x_packed: x_packed should be a 2-tuple: (embedding, reset vector = x-mask)
    :return: updated hidden layer and reset hidden layer
    """

    # Unpack mandatory packed force_reset_vector, x = embedding
    x, reset_vector = x_packed

    with tf.variable_scope(name):
        h = _gru_layer(h_prev, x, 'gru', x_dim, y_dim, reuse=reuse)

        # Force reset hidden state: is set to zero if reset vector consists of zeros
        h_reset = reset_vector * h

    return tf.pack([h, h_reset])


def gru_layer_with_retain(h_prev, x_packed, name='gru', x_dim=256, y_dim=512, reuse=None):
    """
    Used for the session encoder layer. The current state of the session encoder
    should be retained if no EoQ symbol has been reached yet.
    :param h_prev: previous state of the GRU layer
    :param x_packed: x_packed should be a 2-tuple (embedding, retain vector = x-mask)
    """

    # Unpack mandatory packed retain_vector
    x, retain_vector = x_packed

    with tf.variable_scope(name):
        h = _gru_layer(h_prev, x, 'gru', x_dim, y_dim, reuse=reuse)

        # Force reset hidden state: is h_prev is retain vector consists of ones,
        # is h if retain vector consists of zeros
        h_retain = retain_vector * h_prev + tf.sub(np.float32(1.0), retain_vector) * h

    return tf.pack([h, h_retain])


def gru_layer_with_state_reset(h_prev, x_packed, name='gru', x_dim=256, h_dim=512, y_dim=1024, reuse=None):
    """
    Used for the decoder layer
    :param h_prev: previous decoder state
    :param x_packed: should be a 3-tuple (embedder, mask, session_encoder)
    """

    # h_prev = tf.Print(h_prev, [h_prev[:, 1, :]], message="hidden_query: ", summarize=20)

    # Unpack mandatory packed retain_vector and the state
    # x = embedder, ratain_vector = mask, state = session_encoder
    x, retain_vector, state = x_packed

    with tf.variable_scope(name):

        with tf.variable_scope('state_start', reuse=reuse):
            W = tf.get_variable(name='weight', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))

            h_prev_state = retain_vector * h_prev + tf.sub(np.float32(1.0), retain_vector) * tf.tanh(tf.matmul(state, W) + b)

        h = _gru_layer(h_prev_state, x, 'gru', x_dim, y_dim, reuse=reuse)

    return h


def output_layer(x, h, name='output', x_dim=256, y_dim=512, h_dim=512, reuse=None):
    """
    Used after the decoder
    This is used for "full" state bias in the decoder which we did not use in the end.
    """

    with tf.variable_scope(name, reuse=reuse):
        Wh = tf.get_variable(name='weight_hidden', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        Wi = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='bias_input', shape=(y_dim,), initializer=tf.random_normal_initializer(stddev=0.01))

        y = tf.matmul(h, Wh) \
            + tf.matmul(x, Wi) \
            + b

    return y


def output_layer_with_state_bias(x, h, state, name='output', x_dim=256, y_dim=512, h_dim=512, s_dim=512, reuse=None):
    """
    Used after the decoder
    This is used for "full" state bias in the decoder which we did not use in the end.
    """

    with tf.variable_scope(name, reuse=reuse):
        Wh = tf.get_variable(name='weight_hidden', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        Ws = tf.get_variable(name='weight_state', shape=(s_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        Wi = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='bias_input', shape=(y_dim,), initializer=tf.random_normal_initializer(stddev=0.01))

        y = tf.matmul(h, Wh) \
            + tf.matmul(state, Ws) \
            + tf.matmul(x, Wi) \
            + b

    return y


def logits_layer(x, name='logits', x_dim=512, y_dim=90004, reuse=None):
    """
    Used to compute the logits after the output layer.
    The logits could be fed to a softmax layer

    :param x: output (obtained in layers.output_layer)
    :return: logits
    """

    with tf.variable_scope(name, reuse=reuse):

        W = tf.get_variable(name='weight', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.random_normal_initializer(stddev=0.01))

        y = tf.matmul(x, W) + b

    return y


def _gru_layer(h_prev, x, name='gru', x_dim=256, y_dim=512, reuse=None):
    """
    Used for both encoder layers
    """

    with tf.variable_scope(name):

        # Reset gate
        with tf.variable_scope('reset_gate', reuse=reuse):
            Wi_r = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_r = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            b_r = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            r = tf.sigmoid(tf.matmul(x, Wi_r) + tf.matmul(h_prev, Wh_r) + b_r)

        # Update gate
        with tf.variable_scope('update_gate', reuse=reuse):
            Wi_z = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_z = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            b_z = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            z = tf.sigmoid(tf.matmul(x, Wi_z) + tf.matmul(h_prev, Wh_z) + b_z)

        # Candidate update
        with tf.variable_scope('candidate_update', reuse=reuse):
            Wi_h_tilde = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_h_tilde = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            h_tilde = tf.tanh(tf.matmul(x, Wi_h_tilde) + tf.matmul(r * h_prev, Wh_h_tilde) + b_h_tilde)

        # Final update
        h = tf.sub(np.float32(1.0), z) * h_prev + z * h_tilde

    return h


def _rnn_layer(h_prev, x, name='rnn', x_dim=256, y_dim=512, reuse=None):
    """
    Used for both encoder layers,
    this was used for debug purposes
    """

    with tf.variable_scope(name, reuse=reuse):

        Wi = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        Wh = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
        b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))

        h = tf.tanh(tf.matmul(x, Wi) + tf.matmul(h_prev, Wh) + b)

    return h


def _gru_layer_with_state_bias(h_prev, x, state, name='gru', x_dim=256, y_dim=1024, s_dim=512, reuse=None):
    """
    Used for decoder. In this GRU the state of the session encoder layer is used when
    computing the decoder updates.
    This is used for "full" state bias in the decoder which we did not use in the end.
    """

    with tf.variable_scope(name):

        # Reset gate
        with tf.variable_scope('reset_gate', reuse=reuse):
            Wi_r = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_r = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            Ws_r = tf.get_variable(name='weight_state', shape=(s_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            b_r = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            r = tf.sigmoid(tf.matmul(x, Wi_r) + tf.matmul(h_prev, Wh_r) + tf.matmul(state, Ws_r) + b_r)

        # Update gate
        with tf.variable_scope('update_gate', reuse=reuse):
            Wi_z = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_z = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            Ws_z = tf.get_variable(name='weight_state', shape=(s_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            b_z = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            z = tf.sigmoid(tf.matmul(x, Wi_z) + tf.matmul(h_prev, Wh_z) + tf.matmul(state, Ws_z) + b_z)

        # Candidate update
        with tf.variable_scope('candidate_update', reuse=reuse):
            Wi_h_tilde = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_h_tilde = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            Ws_h_tilde = tf.get_variable(name='weight_state', shape=(s_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            h_tilde = tf.tanh(tf.matmul(x, Wi_h_tilde) + \
                      tf.matmul(r * h_prev, Wh_h_tilde) + \
                      tf.matmul(state, Ws_h_tilde) + \
                      b_h_tilde)

        # Final update
        h = tf.sub(np.float32(1.0), z) * h_prev + z * h_tilde

    return h
