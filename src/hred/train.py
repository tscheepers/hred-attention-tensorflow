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

CHECKPOINT_FILE = '../../checkpoints/model-1.ckpt'

DATA_FILE = TRAIN_FILE
SAMPLE_FILE = '../../data/sample_aol_sess_50000.out'

if __name__ == '__main__':

    with tf.Graph().as_default():

        hred = HRED()
        batch_size = None
        max_length = None

        X = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        Y = tf.placeholder(tf.int64, shape=(max_length, batch_size))

        X_beam = tf.placeholder(tf.int64, shape=(batch_size,))
        H_query = tf.placeholder(tf.float32, shape=(batch_size, hred.query_hidden_size))
        H_session = tf.placeholder(tf.float32, shape=(batch_size, hred.session_hidden_size))
        H_decoder = tf.placeholder(tf.float32, shape=(batch_size, hred.decoder_hidden_size))

        logits = hred.step_through_session(X)
        loss = hred.loss(X, logits, Y)
        softmax = hred.softmax(logits)
        accuracy = hred.non_padding_accuracy(logits, Y)
        non_symbol_accuracy = hred.non_symbol_accuracy(logits, Y)

        session_inference = hred.step_through_session(X, return_softmax=True, return_last_with_hidden_states=True, reuse=True)
        step_inference = hred.single_step(X_beam, H_query, H_session, H_decoder, reuse=True)

        optimizer = Optimizer(loss, initial_learning_rate=0.0002, num_steps_per_decay=10000,
                      decay_rate=0.5, max_global_norm=1.0)

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

                    # Accumulative cost, like in hred-qs
                    total_loss += loss_out
                    n_pred += seq_len * batch_size
                    cost = total_loss / n_pred

                    print("Step %d - Cost: %f   Loss: %f   Accuracy: %f   Accuracy (no symbols): %f" %
                          (iteration, cost, loss_out, acc_out, accuracy_non_special_symbols_out))

                    if iteration % 100 == 0:

                        if not math.isnan(loss_out):
                            # Save the variables to disk.
                            save_path = saver.save(sess, CHECKPOINT_FILE)
                            print("Model saved in file: %s" % save_path)

                            read_data.read_batch(
                                DATA_FILE,
                                batch_size=batch_size,
                                max_seq_len=max_length
                            )

                        for (x, _) in read_data.read_line(SAMPLE_FILE):

                            input_x = np.expand_dims(np.asarray(x), 1)

                            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                                session_inference,
                                hred.populate_feed_dict_with_defaults(
                                     batch_size=1,
                                     feed_dict={X: input_x}
                                )
                            )

                            x = np.argmax(softmax_out, axis=1)
                            result = [x]

                            i = 0
                            max_i = 30

                            while x != hred.eos_symbol and i < max_i:
                                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                                    step_inference,
                                    {X_beam: x, H_query: hidden_query, H_session: hidden_session, H_decoder: hidden_decoder}
                                )
                                result += [np.argmax(softmax_out, axis=1)]
                                i += 1

                            input_x = np.array(input_x).flatten()
                            result = np.array(result).flatten()

                            print('Sample input: %s' % (' '.join(map(str, input_x)),))
                            print('Sample output: %s' % (' '.join(map(str, result)),))

                    iteration += 1
