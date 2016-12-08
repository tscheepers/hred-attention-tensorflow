""" File to build and train the entire computation graph in tensorflow
"""

import numpy as np
import tensorflow as tf
import subprocess

tf.logging.set_verbosity(tf.logging.DEBUG)

from hred import HRED
from optimizer import Optimizer
import cPickle
import math
import allesandro.data_iterator as allesandro_data_iterator

VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
TRAIN_DIR = 'logs'

CHECKPOINT_FILE = '../../checkpoints/model-large.ckpt'
VOCAB_FILE = '../../data/aol_vocab_50000.pkl'
TRAIN_FILE = '../../data/aol_sess_50000.out'
SAMPLE_FILE = '../../data/sample_aol_sess_50000.out'
VOCAB_SIZE = 50003
EMBEDDING_DIM = 25
QUERY_DIM = 50
SESSION_DIM = 100
BATCH_SIZE = 80
MAX_LENGTH = 50

# CHECKPOINT_FILE = '../../checkpoints/model-small.ckpt'
# VOCAB_FILE = '../../data/aol_vocab_2500.pkl'
# TRAIN_FILE = '../../data/small_train.out'
# SAMPLE_FILE = '../../data/sample_small_train.out'
# VOCAB_SIZE = 2504
# EMBEDDING_DIM = 10
# QUERY_DIM = 15
# SESSION_DIM = 20
# BATCH_SIZE = 80
# MAX_LENGTH = 50
SEED = 1234


class Trainer(object):

    def __init__(self):
        self.vocab_lookup_dict = cPickle.load(open("../../data/allesandro/all/train.dict.pkl", 'r'))
        self.train_data, _ = allesandro_data_iterator.get_batch_iterator(np.random.RandomState(SEED), {
            'eoq_sym': 1,
            'eos_sym': 2,
            'sort_k_batches': 20,
            'bs': BATCH_SIZE,
            'train_session': "../../data/allesandro/all/train.ses.pkl",
            'train_rank': "../../data/allesandro/all/train.rnk.pkl",
            'seqlen': MAX_LENGTH,
            'valid_session': "../../data/allesandro/all/valid.ses.pkl",
            'valid_rank': "../../data/allesandro/all/valid.rnk.pkl"
        })
        self.train_data.start()

        vocab_size = len(self.vocab_lookup_dict)
        # vocab_size = VOCAB_SIZE

        # self.vocab_lookup_dict = read_data.read_vocab_lookup(VOCAB_FILE)
        self.hred = HRED(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, query_dim=QUERY_DIM,
                         session_dim=SESSION_DIM, decoder_dim=QUERY_DIM, output_dim=EMBEDDING_DIM)

        batch_size = None
        max_length = None

        self.X = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        self.Y = tf.placeholder(tf.int64, shape=(max_length, batch_size))

        # self.X_sample = tf.placeholder(tf.int64, shape=(batch_size,))
        # self.H_query = tf.placeholder(tf.float32, shape=(batch_size, self.hred.query_dim))
        # self.H_session = tf.placeholder(tf.float32, shape=(batch_size, self.hred.session_dim))
        # self.H_decoder = tf.placeholder(tf.float32, shape=(batch_size, self.hred.decoder_dim))

        self.logits = self.hred.step_through_session(self.X)
        self.loss = self.hred.loss(self.X, self.logits, self.Y)
        self.softmax = self.hred.softmax(self.logits)
        self.accuracy = self.hred.non_padding_accuracy(self.logits, self.Y)
        self.non_symbol_accuracy = self.hred.non_symbol_accuracy(self.logits, self.Y)

        # self.session_inference = self.hred.step_through_session(self.X, return_softmax=True,
        #                                                         return_last_with_hidden_states=True, reuse=True)
        # self.step_inference = self.hred.single_step(self.X_sample, self.H_query, self.H_session, self.H_decoder,
        #                                             reuse=True)

        self.optimizer = Optimizer(self.loss)
        self.summary = tf.merge_all_summaries()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()



    def train(self, max_epochs=1000, max_length=50, batch_size=80):

        # Add an op to initialize the variables.
        init_op = tf.initialize_all_variables()

        with tf.Session() as tf_sess:

            tf_sess.run(init_op)
            summary_writer = tf.train.SummaryWriter(TRAIN_DIR, tf_sess.graph)

            total_loss = 0.0
            n_pred = 0.0

            for iteration in range(10000):

                data = self.train_data.next()

                # The training is done with a trick. We append a special </q> at the beginning of the dialog
                # so that we can predict also the first sent in the dialog starting from the dialog beginning token (</q>).
                seq_len = data['max_length']
                prepend = np.ones((1, data['x'].shape[1]))
                x_data_full = np.concatenate((prepend, data['x']))
                x_batch = x_data_full[:seq_len]
                y_batch = x_data_full[1:seq_len + 1]

                # x_batch = np.transpose(np.asarray(x_batch))
                # y_batch = np.transpose(np.asarray(y_batch))

                loss_out, _, acc_out, accuracy_non_special_symbols_out = tf_sess.run(
                    [self.loss, self.optimizer.optimize_op, self.accuracy, self.non_symbol_accuracy],
                    self.hred.populate_feed_dict_with_defaults(
                        batch_size=batch_size, feed_dict={self.X: x_batch, self.Y: y_batch}
                    )
                )

                # Sumerize
                if iteration % 100 == 0:
                    summary_str = tf_sess.run(self.summary, self.hred.populate_feed_dict_with_defaults(
                        batch_size=batch_size, feed_dict={self.X: x_batch, self.Y: y_batch}
                    ))
                    summary_writer.add_summary(summary_str, iteration)
                    summary_writer.flush()

                # Accumulative cost, like in hred-qs
                total_loss += loss_out
                n_pred += seq_len * batch_size
                cost = total_loss / n_pred

                if iteration % 10 == 0:
                    print("Step %d - Cost: %f   Loss: %f   Accuracy: %f   Accuracy (no symbols): %f  Length: %d" %
                      (iteration, cost, loss_out, acc_out, accuracy_non_special_symbols_out, seq_len))

                if iteration % 100 == 0:
                    self.save_model(tf_sess, loss_out)
                    self.sample(tf_sess)

                iteration += 1

    def sample(self, sess, max_sample_length=30):

        # for (x, _) in read_data.read_line(SAMPLE_FILE):
        #
        #     input_x = np.expand_dims(np.asarray(x), 1)
        #
        #     softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
        #         self.session_inference,
        #         self.hred.populate_feed_dict_with_defaults(
        #             batch_size=1, feed_dict={self.X: input_x}
        #         )
        #     )
        #
        #     x = np.argmax(softmax_out, axis=1)
        #     result = [x]
        #     i = 0
        #
        #     while x != self.hred.eos_symbol and i < max_sample_length:
        #         softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
        #             self.step_inference,
        #             {self.X_sample: x, self.H_query: hidden_query, self.H_session: hidden_session,
        #              self.H_decoder: hidden_decoder}
        #         )
        #
        #         x = np.argmax(softmax_out, axis=1)
        #         result += [x]
        #         i += 1
        #
        #     input_x = np.array(input_x).flatten()
        #     result = np.array(result).flatten()
        #     print('Sample input: %s' % (' '.join([self.vocab_lookup_dict.get(x, "NOT_FOUND") for x in input_x]),))
        #     print('Sample output words: %s' % (' '.join([self.vocab_lookup_dict.get(x, "NOT_FOUND") for x in result])))
        pass

    def save_model(self, sess, loss_out):
        if not math.isnan(loss_out):
            # Save the variables to disk.
            save_path = self.saver.save(sess, CHECKPOINT_FILE)
            print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    with tf.Graph().as_default():
        trainer = Trainer()
        trainer.train(batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
