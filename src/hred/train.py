import numpy as np

from hred import HRED
import tensorflow as tf

if __name__ == '__main__':
    with tf.Graph().as_default():

        hred = HRED()
        batch_size = None

        X = tf.placeholder(tf.int32, shape=(batch_size, 1))
        graph = hred.step(X)

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())

            summary_writer = tf.train.SummaryWriter('logs/graph', sess.graph)

            x = np.random.randint(0, hred.vocab_size, (5, 1))

            print(x)

            softmax, output, decoder, session_encoder, query_encoder = sess.run(graph, {X: x})

            print(softmax)