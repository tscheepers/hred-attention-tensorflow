from hred import HRED
import tensorflow as tf

if __name__ == '__main__':

    hred = HRED()
    batch_size = None

    X = tf.placeholder(tf.int32, shape=(batch_size, hred.vocab_size))
    graph = hred.step(X)

    with tf.Session() as sess:

        summary_writer = tf.train.SummaryWriter('logs/graph', sess.graph)

        sess.run(graph)