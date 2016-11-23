import tensorflow as tf
import numpy as np

from hred.initializers import orthogonal_initializer


def embedding_layer(x, name='embedding-layer', vocab_dim=10000, embedding_dim=256):

    with tf.variable_scope(name):
        W = tf.get_variable(name="weights", shape=(vocab_dim, embedding_dim),
                            initializer=tf.truncated_normal)
        embedding = tf.nn.embedding_lookup(W, x)

    return embedding


def gru_layer_with_reset(h_prev, x, name='gru', x_dim=256, h_dim=512):

    # Unpack mandatory packed force_reset_vector
    force_reset_vector, x = tf.unpack(0, x)

    with tf.variable_scope(name):
        h = gru_layer(h_prev, x, name, x_dim, h_dim)

        # Force reset hidden state
        h_reset = force_reset_vector * h

    return h, h_reset


def gru_layer(h_prev, x, name='gru', x_dim=256, h_dim=512):

    with tf.variable_scope(name):

        # Reset gate
        with tf.variable_scope('reset_gate'):
            Wi_r = tf.get_variable(name='weight_input', shape=(x_dim, h_dim), initializer=tf.truncated_normal)
            Wh_r = tf.get_variable(name='weight_hidden', shape=(h_dim, h_dim), initializer=orthogonal_initializer())
            b_r = tf.get_variable(name='bias', shape=(h_dim,), initializer=tf.zeros)
            r = tf.sigmoid(tf.matmul(x, Wi_r)) + tf.matmul(h_prev, Wh_r) + b_r

        # Update gate
        with tf.variable_scope('update_gate'):
            Wi_z = tf.get_variable(name='weight_input', shape=(x_dim, h_dim), initializer=tf.truncated_normal)
            Wh_z = tf.get_variable(name='weight_hidden', shape=(h_dim, h_dim), initializer=orthogonal_initializer())
            b_z = tf.get_variable(name='bias', shape=(h_dim,), initializer=tf.zeros)
            z = tf.sigmoid(tf.matmul(x, Wi_z)) + tf.matmul(h_prev, Wh_z) + b_z

        # Candidate update
        with tf.variable_scope('candidate_update'):
            Wi_h_tilde = tf.get_variable(name='weight_input', shape=(x_dim, h_dim), initializer=tf.truncated_normal)
            Wh_h_tilde = tf.get_variable(name='weight_hidden', shape=(h_dim, h_dim), initializer=orthogonal_initializer())
            b_h_tilde = tf.get_variable(name='bias', shape=(h_dim,), initializer=tf.zeros)
            h_tilde = tf.tanh(tf.matmul(x, Wi_h_tilde)) + tf.matmul(r * h_prev, Wh_h_tilde) + b_h_tilde

        # Final update
        h = (np.float32(1.0) - z) * h_prev + z * h_tilde

    return h

