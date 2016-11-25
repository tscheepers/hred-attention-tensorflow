import numpy as np

from hred import HRED
import tensorflow as tf

if __name__ == '__main__':
    with tf.Graph().as_default():

        hred = HRED()
        batch_size = None
        max_length = None

        X = tf.placeholder(tf.int32, shape=(max_length, batch_size))
        HQ0 = tf.placeholder(tf.float32, (2, batch_size, hred.query_hidden_size))
        HS0 = tf.placeholder(tf.float32, (2, batch_size, hred.session_hidden_size))
        HD0 = tf.placeholder(tf.float32, (batch_size, hred.decoder_hidden_size))
        O0 = tf.placeholder(tf.float32, (batch_size, hred.vocab_size))

        graph = hred.step(X, HQ0, HS0, HD0, O0)

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())

            batch_size = 10
            max_length = 5

            summary_writer = tf.train.SummaryWriter('logs/graph', sess.graph)

            x = np.random.randint(0, hred.vocab_size, (max_length, batch_size))
            hq0 = np.zeros((2, batch_size, hred.query_hidden_size))
            hs0 = np.zeros((2, batch_size, hred.session_hidden_size))
            hd0 = np.zeros((batch_size, hred.decoder_hidden_size))
            o0 = np.zeros((batch_size, hred.vocab_size))

            print(x)

            softmax = sess.run(
                graph,
                {X: x, HQ0: hq0, HS0: hs0, HD0: hd0, O0: o0}
            )

            print(softmax)