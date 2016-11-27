""" File to build and train the entire computation graph in tensorflow
"""

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)

from hred import HRED
from optimizer import Optimizer
import read_data

TRAIN_FILE = '../../data/tr_session.out'
VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
SMALL_FILE = '../../data/small_test_session.out'
eos = '2'

TRAIN_TFR = '../../data/tfrecords/train.tfrecords'
VALIDATION_TFR = '../../data/tfrecords/valid.tfrecords'
TEST_TFR = '../../data/tfrecords/test.tfrecords'
SMALL_TFR = '../../data/tfrecords/small.tfrecords'

DATA_FILE = TRAIN_FILE

if __name__ == '__main__':

    with tf.Graph().as_default():

        hred = HRED()
        batch_size = None
        max_length = None

        X = tf.placeholder(tf.int32, shape=(max_length, batch_size))
        Y = tf.placeholder(tf.int32, shape=(max_length, batch_size))

        # STUFF for pipeline
        # input, label = read_data.read_and_decode(SMALL_TFR)
        #
        # input_batch, labels_batch = tf.train.shuffle_batch(
        #     [input, label], batch_size=1,
        #     capacity=2000,
        #     min_after_dequeue=1000)
        #
        # print input_batch.values

        logits = hred.step_through_session(X)
        loss = hred.loss(logits, Y)
        softmax = hred.softmax(logits)

        optimizer = Optimizer(loss, initial_learning_rate=1e-2, max_global_norm=1.0)
        optimize = optimizer.optimize_op

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter('logs/graph', sess.graph)

            batch_size = 80#80 #200#10 # 200 #10
            max_length = 5
            iterations = 1 #6 # 100


            # TODO: This really is an ugly way to read in the data, we should really really change this
            idx = 0
            x_batch = []
            y_batch = []

            for (x, y) in read_data.read_data(DATA_FILE):
                if idx == batch_size:
                    # do your stuff

                    x_batch = np.transpose(np.asarray(x_batch))
                    y_batch = np.transpose(np.asarray(y_batch))
                    # print "idx", idx
                    # print "x", x_batch
                    # print "y", y_batch

                    loss_out, _, softmax_out = sess.run(
                        [loss, optimize, softmax],
                        hred.populate_feed_dict_with_defaults(
                            batch_size=batch_size,
                            feed_dict={X: x_batch, Y: y_batch}
                        )
                    )
                    print("Loss: %f" % loss_out)
                    print("Softmax", np.argmax(softmax_out, axis=2))

                    # and reset
                    idx = 0
                    x_batch = []
                    y_batch = []
                else:
                    x_batch.append(x)
                    y_batch.append(y)
                    idx += 1












# for x in range(iterations):
#
#     r = np.random.randint(0, hred.vocab_size, (max_length + 1, batch_size))
#     x = r[:-1,:]
#     y = r[1:,:]
#     hq0 = np.zeros((2, batch_size, hred.query_hidden_size))
#     hs0 = np.zeros((2, batch_size, hred.session_hidden_size))
#     hd0 = np.zeros((batch_size, hred.decoder_hidden_size))
#     o0 = np.zeros((batch_size, hred.output_hidden_size))
#     l0 = np.zeros((batch_size, hred.vocab_size))
#
#     print(r)
#     print(x)
#     print(y)
#
#     loss_out, _ = sess.run(
#         [loss, optimize],
#         {X: x, Y:y, HQ0: hq0, HS0: hs0, HD0: hd0, O0: o0, L0: l0}
#     )
#
#     print("Loss: %f" % loss_out)
#
#
