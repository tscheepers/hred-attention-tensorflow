""" File to build and train the entire computation graph in tensorflow
"""

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)

from hred import HRED
from optimizer import Optimizer
import read_data
import math

TRAIN_FILE = '../../data/aol_sess_50000.out'
VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
SMALL_FILE = '../../data/small_train.out'

CHECKPOINT_FILE = '../../checkpoints/model-50000.ckpt'

DATA_FILE = TRAIN_FILE # TRAIN_FILE

if __name__ == '__main__':

    with tf.Graph().as_default():

        hred = HRED()
        batch_size = None
        max_length = None

        X = tf.placeholder(tf.int32, shape=(max_length, batch_size))
        Y = tf.placeholder(tf.int32, shape=(max_length, batch_size))

        logits = hred.step_through_session(X)
        loss = hred.loss(X, logits, Y)
        softmax = hred.softmax(logits)
        accuracy = hred.accuracy(logits, Y)

        optimizer = Optimizer(loss, learning_rate=0.0002, max_global_norm=1.0)

        # Add an op to initialize the variables.
        init_op = tf.initialize_all_variables()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(init_op)

            # summary_writer = tf.train.SummaryWriter('logs/graph', sess.graph)

            batch_size = 80
            max_length = 50
            max_iterations = 1
            max_epochs = 1000
            iteration = 0

            for epoch in range(max_epochs):

                for ((x_batch, y_batch), seq_len) in read_data.read_batch(
                        DATA_FILE,
                        batch_size=batch_size,
                        max_seq_len=max_length
                ):
                    x_batch = np.transpose(np.asarray(x_batch))
                    y_batch = np.transpose(np.asarray(y_batch))

                    # print "x", x_batch
                    # print "y", y_batch
                    # print "seq len", seq_len

                    loss_out, _, softmax_out, acc_out = sess.run(
                        [loss, optimizer.optimize_op, softmax, accuracy],
                        hred.populate_feed_dict_with_defaults(
                            batch_size=batch_size,
                            feed_dict={X: x_batch, Y: y_batch}
                        )
                    )

                    if iteration % 10 == 0:
                        print("Loss %d: %f" % (iteration, loss_out))
                        print("Acc %d: %f" % (iteration, acc_out))

                    if iteration % 100 == 0:
                        print("Input", x_batch)
                        print("Softmax", np.argmax(softmax_out, axis=2))

                        if not math.isnan(loss_out):
                            # Save the variables to disk.
                            save_path = saver.save(sess, CHECKPOINT_FILE)
                            print("Model saved in file: %s" % save_path)

                    iteration += 1
