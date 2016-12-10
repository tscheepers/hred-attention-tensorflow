import tensorflow as tf
import numpy as np

# From allessandro
# # Dimensionality of hidden layers
# state['qdim'] = 512
# # Dimensionality of session hidden layer
# state['sdim'] = 1024
# # Dimensionality of low-rank approximation
# state['rankdim'] = 256
# state['lambda_rank'] = 0.

RANKDIM = 256
QDIM = 512 # dimensionality of hidden layers (q for query)
SDIM = 1024 # dimensionality of session hidden layers


def gated_query_step(x_t, m_t, hr_tm1):

    # Embedding
    W_in = tf.get_variable(name='W_in',
                             shape=[RANKDIM, QDIM],
                             initializer=tf.truncated_normal,
                             regularizer=None)

    # TODO: orthogonal init (see below)
    W_hh = tf.get_variable(name='W_in',
                             shape=[QDIM, QDIM],
                             initializer=tf.truncated_normal,
                             regularizer=None)

    b_hh = tf.get_variable(name='b_z',
                             shape=[QDIM,],
                             initializer=tf.zeros,
                             regularizer=None)

    # Gated
    W_in_r = tf.get_variable(name='W_in_r',
                             shape=[RANKDIM, QDIM],
                             initializer=tf.truncated_normal,
                             regularizer=None)

    W_in_z = tf.get_variable(name='W_in_z',
                             shape=[RANKDIM, QDIM],
                             initializer=tf.truncated_normal,
                             regularizer=None)

    # TODO: initalizer is actually a orthogonal init --> guess we need to build it ourselves:
    # http://stats.stackexchange.com/questions/228704/how-does-one-initialize-neural-networks-as-suggested-by-saxe-et-al-using-orthogo
    W_hh_r = tf.get_variable(name='W_hh_r',
                             shape=[QDIM, QDIM],
                             initializer=tf.truncated_normal,
                             regularizer=None)

    # TODO: again orthogonal initialization
    W_hh_z = tf.get_variable(name='W_hh_z',
                             shape=[QDIM, QDIM],
                             initializer=tf.truncated_normal,
                             regularizer=None)

    b_z = tf.get_variable(name='b_z',
                             shape=[QDIM,],
                             initializer=tf.zeros,
                             regularizer=None)


    b_r = tf.get_variable(name='b_r',
                             shape=[QDIM, ],
                             initializer=tf.zeros,
                             regularizer=None)

    # TODO: what is this dimshuffle?

    # Reset gate
    r_t = tf.sigmoid(tf.matmul(x_t, W_in_r)) + tf.matmul(hr_tm1, W_hh_r) + b_r

    # Update gate
    z_t = tf.sigmoid(tf.matmul(x_t, W_in_z)) + tf.matmul(hr_tm1, W_hh_z) + b_z

    # Candidate update
    h_tilde = tf.tanh(tf.matmul(x_t, W_in)) + tf.matmul(r_t * hr_tm1, W_hh) + b_hh

    # Final update
    h_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde

    # Reset gate
    hr_t = m_t * h_t

    return h_t, hr_t, r_t, z_t, h_tilde


def gated_session_step()


