import tensorflow as tf


class Optimizer(object):

    def __init__(self, loss, learning_rate, max_global_norm=1.0):
        """ Create a simple optimizer.

        This optimizer clips gradients and uses vanilla stochastic gradient
        descent with a learning rate that decays exponentially.

        Args:
            loss: A 0-D float32 Tensor.
            learning_rate: A float.
            max_global_norm: A float. If the global gradient norm is less than
                this, do nothing. Otherwise, rescale all gradients so that
                the global norm because `max_global_norm`.
        """

        trainables = tf.trainable_variables()
        grads = tf.gradients(loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.1, beta2=0.001)
        self._optimize_op = optimizer.apply_gradients(grad_var_pairs, global_step=self.global_step)

    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op
