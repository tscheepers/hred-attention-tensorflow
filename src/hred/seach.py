""" File to build to perform beamsearch for the next query
"""

import numpy as np
import tensorflow as tf

from hred import HRED

tf.logging.set_verbosity(tf.logging.DEBUG)


CHECKPOINT_FILE = '../../checkpoints/model.ckpt'

# For beam search example:
# https://github.com/tensorflow/tensorflow/issues/654#issuecomment-168237741

if __name__ == '__main__':
    with tf.Graph().as_default():
        hred = HRED()
        batch_size = None
        max_length = None

        X = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        X_beam = tf.placeholder(tf.int64, shape=(batch_size, ))
        H_query = tf.placeholder(tf.float32, shape=(batch_size, hred.query_hidden_size))
        H_session = tf.placeholder(tf.float32, shape=(batch_size, hred.session_hidden_size))
        H_decoder = tf.placeholder(tf.float32, shape=(batch_size, hred.decoder_hidden_size))

        session_result = hred.step_through_session(X, return_last_with_hidden_states=True, return_softmax=True)
        step_result = hred.single_step(X_beam, H_query, H_session, H_decoder)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:

            # Restore variables from disk.
            saver.restore(sess, CHECKPOINT_FILE)
            print("Model restored.")

            x = np.array([10, 11, 12, hred.eoq_symbol, 13, 14, 15, hred.eoq_symbol, 16, 17, 18, hred.eoq_symbol])
            x = np.expand_dims(x, 1)
            batch_size = 1

            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(session_result,
                hred.populate_feed_dict_with_defaults(
                    batch_size=batch_size,
                    feed_dict={X: x}
                )
            )

            x = np.argmax(softmax_out, axis=1)
            print(x)

            i = 0
            max_i = 100

            while x != hred.eoq_symbol and i < max_i:

                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                    step_result,
                    {X_beam: x, H_query: hidden_query, H_session: hidden_session, H_decoder: hidden_decoder}
                )

                x = np.argmax(softmax_out, axis=1)
                print(x)

                i += 1