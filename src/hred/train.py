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
TRAIN_DIR = 'logs'

CHECKPOINT_FILE = '../../checkpoints/model-1.ckpt'

DATA_FILE = TRAIN_FILE

if __name__ == '__main__':

    with tf.Graph().as_default():

        hred = HRED()
        batch_size = None
        max_length = None

        X = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        Y = tf.placeholder(tf.int64, shape=(max_length, batch_size))

        logits = hred.step_through_session(X)
        loss = hred.loss(X, logits, Y)
        softmax = hred.softmax(logits)
        accuracy = hred.non_padding_accuracy(logits, Y)
        non_symbol_accuracy = hred.non_symbol_accuracy(logits, Y)

        optimizer = Optimizer(loss, initial_learning_rate=0.0002, num_steps_per_decay=1000,
                              decay_rate=0.5, max_global_norm=1.0)

        summary = tf.merge_all_summaries()

        # Add an op to initialize the variables.
        init_op = tf.initialize_all_variables()

        # Add ops to save and restore all the variables.

        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(init_op)
            summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)


            batch_size = 80
            max_length = 50
            max_iterations = 1
            max_epochs = 1000
            iteration = 0

            total_loss = 0.0
            n_pred = 0.0

            for epoch in range(max_epochs):

                for ((x_batch, y_batch), seq_len) in read_data.read_batch(
                        DATA_FILE,
                        batch_size=batch_size,
                        max_seq_len=max_length
                ):
                    x_batch = np.transpose(np.asarray(x_batch))
                    y_batch = np.transpose(np.asarray(y_batch))

                    loss_out, _, softmax_out, acc_out, accuracy_non_special_symbols_out = sess.run(
                        [loss, optimizer.optimize_op, softmax, accuracy, non_symbol_accuracy],
                        hred.populate_feed_dict_with_defaults(
                            batch_size=batch_size,
                            feed_dict={X: x_batch, Y: y_batch}
                        )
                    )

                    summary_str = sess.run(summary, hred.populate_feed_dict_with_defaults(
                        batch_size=batch_size,
                        feed_dict={X: x_batch, Y: y_batch}
                    ))
                    summary_writer.add_summary(summary_str, iteration)
                    summary_writer.flush()

                    # Accumulative cost, like in hred-qs
                    total_loss += loss_out
                    n_pred += seq_len * batch_size
                    cost = total_loss / n_pred

                    print("Step %d - Cost: %f   Loss: %f   Accuracy: %f   Accuracy (no symbols): %f" %
                          (iteration, cost, loss_out, acc_out, accuracy_non_special_symbols_out))


                    if iteration % 1000 == 0:

                        if not math.isnan(loss_out):
                            # Save the variables to disk.
                            save_path = saver.save(sess, CHECKPOINT_FILE)

                            print("Model saved in file: %s" % save_path)

                    iteration += 1
