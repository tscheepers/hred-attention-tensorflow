""" File to build and train the entire computation graph in tensorflow
"""

import numpy as np
import tensorflow as tf
import subprocess

tf.logging.set_verbosity(tf.logging.DEBUG) # test

from hred import HRED
from optimizer import Optimizer
import cPickle
import math
import src.sordoni.data_iterator as sordoni_data_iterator

VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
LOGS_DIR = '../../logs'
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2
RESTORE = True

N_BUCKETS = 20

CHECKPOINT_FILE = '../../checkpoints/attention/model-dev_large.ckpt'
# OUR_VOCAB_FILE = '../../data/aol_vocab_50000.pkl'
# OUR_TRAIN_FILE = '../../data/aol_sess_50000.out'
# OUR_SAMPLE_FILE = '../../data/sample_aol_sess_50000.out'
SORDONI_VOCAB_FILE = '../../data/sordoni/dev_large/train.dict.pkl'
SORDONI_TRAIN_FILE = '../../data/sordoni/dev_large/train.ses.pkl'
SORDONI_VALID_FILE = '../../data/sordoni/dev_large/valid.ses.pkl'

# CHECKPOINT_FILE = '../../checkpoints/model-small.ckpt'
# OUR_VOCAB_FILE = '../../data/aol_vocab_2500.pkl'
# OUR_TRAIN_FILE = '../../data/small_train.out'
# OUR_SAMPLE_FILE = '../../data/sample_small_train.out'
# SORDONI_VOCAB_FILE = '../../data/sordoni/dev_large/train.dict.pkl'
# SORDONI_TRAIN_FILE = '../../data/sordoni/dev_large/train.ses.pkl'
# SORDONI_VALID_FILE = '../../data/sordoni/dev_large/valid.ses.pkl'
# VOCAB_SIZE = 2504
# EMBEDDING_DIM = 10
# QUERY_DIM = 15
# SESSION_DIM = 20
# BATCH_SIZE = 80
# MAX_LENGTH = 50
SEED = 1234


class Trainer(object):
    def __init__(self):

        vocab = cPickle.load(open(SORDONI_VOCAB_FILE, 'r'))
        self.vocab_lookup_dict = {k: v for v, k, count in vocab}

        self.train_data, self.valid_data = sordoni_data_iterator.get_batch_iterator(np.random.RandomState(SEED), {
            'eoq_sym': EOQ_SYMBOL,
            'eos_sym': EOS_SYMBOL,
            'sort_k_batches': N_BUCKETS,
            'bs': BATCH_SIZE,
            'train_session': SORDONI_TRAIN_FILE,
            'seqlen': MAX_LENGTH,
            'valid_session': SORDONI_VALID_FILE
        })
        self.train_data.start()
        self.valid_data.start()

        vocab_size = len(self.vocab_lookup_dict)

        # vocab_size = VOCAB_SIZE
        # self.vocab_lookup_dict = read_data.read_vocab_lookup(OUR_VOCAB_FILE)

        self.hred = HRED(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, query_dim=QUERY_DIM,
                         session_dim=SESSION_DIM, decoder_dim=QUERY_DIM, output_dim=EMBEDDING_DIM,
                         eoq_symbol=EOQ_SYMBOL, eos_symbol=EOS_SYMBOL, unk_symbol=UNK_SYMBOL)

        batch_size = None
        max_length = None

        self.X = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        self.Y = tf.placeholder(tf.int64, shape=(max_length, batch_size))

        self.X_sample = tf.placeholder(tf.int64, shape=(batch_size,))
        self.H_query = tf.placeholder(tf.float32, shape=(batch_size, self.hred.query_dim))
        self.H_session = tf.placeholder(tf.float32, shape=(batch_size, self.hred.session_dim))
        self.H_decoder = tf.placeholder(tf.float32, shape=(batch_size, self.hred.decoder_dim))

        self.logits = self.hred.step_through_session(self.X)
        self.loss = self.hred.loss(self.X, self.logits, self.Y)
        self.softmax = self.hred.softmax(self.logits)
        self.accuracy = self.hred.non_padding_accuracy(self.logits, self.Y)
        self.non_symbol_accuracy = self.hred.non_symbol_accuracy(self.logits, self.Y)

        self.session_inference = self.hred.step_through_session(
            self.X, return_softmax=True, return_last_with_hidden_states=True, reuse=True
        )
        self.step_inference = self.hred.single_step(
            self.X_sample, self.H_query, self.H_session, self.H_decoder, reuse=True
        )

        self.optimizer = Optimizer(self.loss)
        self.summary = tf.merge_all_summaries()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def train(self, max_epochs=1000, max_length=50, batch_size=80):

        # Add an op to initialize the variables.
        init_op = tf.initialize_all_variables()

        with tf.Session() as tf_sess:

            tf_sess.run(init_op)
            summary_writer = tf.train.SummaryWriter(LOGS_DIR, tf_sess.graph)

            total_loss = 0.0
            n_pred = 0.0

            for iteration in range(10000):

                x_batch, y_batch, seq_len = self.get_batch(self.train_data)

                if iteration % 10 == 0:
                    loss_out, _, acc_out, accuracy_non_special_symbols_out = tf_sess.run(
                        [self.loss, self.optimizer.optimize_op, self.accuracy, self.non_symbol_accuracy],
                        {self.X: x_batch, self.Y: y_batch}
                    )

                    # Accumulative cost, like in hred-qs
                    total_loss += loss_out
                    n_pred += seq_len * batch_size
                    cost = total_loss / n_pred

                    print("Step %d - Cost: %f   Loss: %f   Accuracy: %f   Accuracy (no symbols): %f  Length: %d" %
                          (iteration, cost, loss_out, acc_out, accuracy_non_special_symbols_out, seq_len))

                else:
                    loss_out, _ = tf_sess.run(
                        [self.loss, self.optimizer.optimize_op],
                        {self.X: x_batch, self.Y: y_batch}
                    )

                    # Accumulative cost, like in hred-qs
                    total_loss += loss_out
                    n_pred += seq_len * batch_size

                # Sumerize
                if iteration % 100 == 0:
                    summary_str = tf_sess.run(self.summary, {self.X: x_batch, self.Y: y_batch})
                    summary_writer.add_summary(summary_str, iteration)
                    summary_writer.flush()

                if iteration % 100 == 0:
                    self.save_model(tf_sess, loss_out)
                    self.sample(tf_sess)

                iteration += 1

    def sample(self, sess, max_sample_length=30, num_of_samples=3):

        for i in range(num_of_samples):

            x_batch, _, seq_len = self.get_batch(self.valid_data)
            input_x = np.expand_dims(x_batch[:-(seq_len / 2), 1], axis=1)

            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                self.session_inference,
                feed_dict={self.X: input_x}
            )

            # Ignore UNK
            arg_sort = np.argsort(softmax_out, axis=1)
            if arg_sort[0, 0] == self.hred.unk_symbol:
                x = arg_sort[:, 1]
            else:
                x = arg_sort[:, 0]

            result = [x]
            i = 0

            while x != self.hred.eos_symbol and i < max_sample_length:
                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                    self.step_inference,
                    {self.X_sample: x, self.H_query: hidden_query, self.H_session: hidden_session,
                     self.H_decoder: hidden_decoder}
                )

                # Ignore UNK
                arg_sort = np.argsort(softmax_out, axis=1)
                if arg_sort[0, 0] == self.hred.unk_symbol:
                    x = arg_sort[:, 1]
                else:
                    x = arg_sort[:, 0]

                result += [x]
                i += 1

            input_x = np.array(input_x).flatten()
            result = np.array(result).flatten()
            print('Sample input:  %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))
            print('Sample output: %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in result])))

    def save_model(self, sess, loss_out):
        if not math.isnan(loss_out):
            # Save the variables to disk.
            save_path = self.saver.save(sess, CHECKPOINT_FILE)
            print("Model saved in file: %s" % save_path)

    def get_batch(self, train_data):

        # The training is done with a trick. We append a special </q> at the beginning of the dialog
        # so that we can predict also the first sent in the dialog starting from the dialog beginning token (</q>).

        data = train_data.next()
        seq_len = data['max_length']
        prepend = np.ones((1, data['x'].shape[1]))
        x_data_full = np.concatenate((prepend, data['x']))
        x_batch = x_data_full[:seq_len]
        y_batch = x_data_full[1:seq_len + 1]

        # x_batch = np.transpose(np.asarray(x_batch))
        # y_batch = np.transpose(np.asarray(y_batch))

        return x_batch, y_batch, seq_len


if __name__ == '__main__':
    with tf.Graph().as_default():
        trainer = Trainer()
        trainer.train(batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
