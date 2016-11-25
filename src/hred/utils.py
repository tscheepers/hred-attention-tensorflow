import numpy as np
import tensorflow as tf


"""
Some code from Keras' TensorFl
"""

def orthogonal_initializer(scale=1.01):
    '''
    Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    # print('Warning -- You have opted to use the orthogonal_initializer function')

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        # print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)

    return _initializer


def dot(x, y):
    '''Multiplies 2 tensors.
    When attempting to multiply a ND tensor
    with a ND tensor, reproduces the Theano behavior
    (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))
    '''
    if ndim(x) > 2:
        x_shape = (-1,) + int_shape(x)[1:]
        y_shape = int_shape(y)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if is_sparse(x):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def ndim(x):
    '''
    Returns the number of axes in a tensor, as an integer.
    '''
    if is_sparse(x):
        return x._dims

    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def int_shape(x):
    '''Returns the shape of a tensor as a tuple of
    integers or None entries.
    Note that this function only works with TensorFlow.
    '''
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])

def is_sparse(tensor):
    return isinstance(tensor, tf.SparseTensor)