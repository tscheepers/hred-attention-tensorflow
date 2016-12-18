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
RESTORE = False

N_BUCKETS = 20
MAX_ITTER = 20


# CHECKPOINT_FILE = '../../checkpoints/model-huge3.ckpt'
# OUR_VOCAB_FILE = '../../data/aol_vocab_50000.pkl'
# OUR_TRAIN_FILE = '../../data/aol_sess_50000.out'
# OUR_SAMPLE_FILE = '../../data/sample_aol_sess_50000.out'
# SORDONI_VOCAB_FILE = '../../data/sordoni/all/train.dict.pkl'
# SORDONI_TRAIN_FILE = '../../data/sordoni/all/train.ses.pkl'
# SORDONI_VALID_FILE = '../../data/sordoni/all/valid.ses.pkl'
# VOCAB_SIZE = 50003
# EMBEDDING_DIM = 25
# QUERY_DIM = 50
# SESSION_DIM = 100
# EMBEDDING_DIM = 128
# QUERY_DIM = 256
# SESSION_DIM = 512
# BATCH_SIZE = 80
# MAX_LENGTH = 50

# CHECKPOINT_FILE = '../../checkpoints/model-small.ckpt'
# OUR_VOCAB_FILE = '../../data/aol_vocab_2500.pkl'
# OUR_TRAIN_FILE = '../../data/small_train.out'
# OUR_SAMPLE_FILE = '../../data/sample_small_train.out'

# SORDONI_VOCAB_FILE = '../../data/sordoni/dev_large/train.dict.pkl'
# SORDONI_TRAIN_FILE = '../../data/sordoni/dev_large/train.ses.pkl'
# SORDONI_VALID_FILE = '../../data/sordoni/dev_large/valid.ses.pkl'

CHECKPOINT_FILE = '../../checkpoints/model-small.ckpt'
OUR_VOCAB_FILE = '../../data/aol_vocab_2500.pkl'
OUR_TRAIN_FILE = '../../data/small_train.out'
OUR_SAMPLE_FILE = '../../data/sample_small_train.out'
SORDONI_VOCAB_FILE = '../../data/sordoni/dev_large/train.dict.pkl'
SORDONI_TRAIN_FILE = '../../data/sordoni/dev_large/train.ses.pkl'
SORDONI_VALID_FILE = '../../data/sordoni/dev_large/valid.ses.pkl'
VOCAB_SIZE = 2504
EMBEDDING_DIM = 10
QUERY_DIM = 15
SESSION_DIM = 20
BATCH_SIZE = 80
MAX_LENGTH = 50
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
        self.H_query = tf.placeholder(tf.float32, shape=(None, batch_size, self.hred.query_dim))
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

            if RESTORE:
                # Restore variables from disk.
                self.saver.restore(tf_sess, CHECKPOINT_FILE)
                print("Model restored.")
            else:
                tf_sess.run(init_op)

            summary_writer = tf.train.SummaryWriter(LOGS_DIR, tf_sess.graph)

            total_loss = 0.0
            n_pred = 0.0

            for iteration in range(MAX_ITTER):

                x_batch, y_batch, seq_len = self.get_batch(self.train_data)

                if iteration % 10 == 0:
                    loss_out, _, acc_out, accuracy_non_special_symbols_out = tf_sess.run(
                        [self.loss, self.optimizer.optimize_op, self.accuracy, self.non_symbol_accuracy],
                        {self.X: x_batch, self.Y: y_batch}
                    )

                    # Accumulative cost, like in hred-qs
                    total_loss_tmp = total_loss + loss_out
                    n_pred_tmp = n_pred + seq_len * batch_size
                    cost = total_loss_tmp / n_pred_tmp

                    print("Step %d - Cost: %f   Loss: %f   Accuracy: %f   Accuracy (no symbols): %f  Length: %d" %
                          (iteration, cost, loss_out, acc_out, accuracy_non_special_symbols_out, seq_len))

                else:
                    loss_out, _ = tf_sess.run(
                        [self.loss, self.optimizer.optimize_op],
                        {self.X: x_batch, self.Y: y_batch}
                    )

                    # Accumulative cost, like in hred-qs
                    total_loss_tmp = total_loss + loss_out
                    n_pred_tmp = n_pred + seq_len * batch_size
                    cost = total_loss_tmp / n_pred_tmp

                if math.isnan(loss_out) or math.isnan(cost) or cost > 100:
                    print("Found inconsistent results, restoring model...")
                    self.saver.restore(tf_sess, CHECKPOINT_FILE)
                else:
                    total_loss = total_loss_tmp
                    n_pred = n_pred_tmp

                    if iteration % 25 == 0:
                        print("Saving..")
                        self.save_model(tf_sess, loss_out)

                # Sumerize
                if iteration % 10 == 0:
                #if iteration % 100 == 0:
                    summary_str = tf_sess.run(self.summary, {self.X: x_batch, self.Y: y_batch})
                    summary_writer.add_summary(summary_str, iteration)
                    summary_writer.flush()

                if iteration % 2 == 0:
                     self.sample(tf_sess)
                #     self.sample_beam(tf_sess)

                iteration += 1

    def sample(self, sess, max_sample_length=30, num_of_samples=3, min_queries = 3):

        for i in range(num_of_samples):

            x_batch, _, seq_len = self.get_batch(self.valid_data)
            input_x = np.expand_dims(x_batch[:-(seq_len / 2), 1], axis=1)

            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                self.session_inference,
                feed_dict={self.X: input_x}
            )

            queries_accepted = 0
            arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]

            # Ignore UNK and EOS (for the first min_queries)
            arg_sort_i = 0
            while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                            arg_sort[arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                arg_sort_i += 1
            x = arg_sort[arg_sort_i]

            if x == self.hred.eoq_symbol:
                queries_accepted += 1

            result = [x]
            i = 0

            while x != self.hred.eos_symbol and i < max_sample_length:
                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                    self.step_inference,
                    {self.X_sample: [x], self.H_query: hidden_query, self.H_session: hidden_session,
                     self.H_decoder: hidden_decoder}
                )
                print("INFO -- Sample hidden states", tf.shape(hidden_query))
                arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]

                # Ignore UNK and EOS (for the first min_queries)
                arg_sort_i = 0
                while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                                arg_sort[arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                    arg_sort_i += 1
                x = arg_sort[arg_sort_i]

                if x == self.hred.eoq_symbol:
                    queries_accepted += 1

                result += [x]
                i += 1

            input_x = np.array(input_x).flatten()
            result = np.array(result).flatten()
            print('Sample input:  %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))
            print('Sample output: %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in result])))
            print('')

    def sample_beam(self, sess, max_steps=30, num_of_samples=3, beam_size=10, min_queries=2):

        for step in range(num_of_samples):

            x_batch, _, seq_len = self.get_batch(self.valid_data)
            input_x = np.expand_dims(x_batch[:-(seq_len / 2), 1], axis=1)

            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                self.session_inference,
                feed_dict={self.X: input_x}
            )

            current_beam_size = beam_size
            current_hypotheses = []
            final_hypotheses = []

            # Reverse arg sort (highest prob above)
            arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]
            arg_sort_i = 0

            # create original current_hypotheses
            while len(current_hypotheses) < current_beam_size:
                # Ignore UNK and EOS (for the first min_queries)
                while arg_sort[arg_sort_i] == self.hred.unk_symbol or arg_sort[arg_sort_i] == self.hred.eos_symbol:
                    arg_sort_i += 1

                x = arg_sort[arg_sort_i]
                arg_sort_i += 1

                queries_accepted = 1 if x == self.hred.eoq_symbol else 0
                result = [x]
                prob = softmax_out[0][x]
                current_hypotheses += [
                    (prob, x, result, hidden_query, hidden_session, hidden_decoder, queries_accepted)]

            # Create hypotheses per step
            step = 0
            while current_beam_size > 0 and step <= max_steps:

                step += 1
                next_hypotheses = []

                # expand all hypotheses
                for prob, x, result, hidden_query, hidden_session, hidden_decoder, queries_accepted in current_hypotheses:

                    softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                        self.step_inference,
                        {self.X_sample: [x], self.H_query: hidden_query, self.H_session: hidden_session,
                         self.H_decoder: hidden_decoder}
                    )

                    # Reverse arg sort (highest prob above)
                    arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]
                    arg_sort_i = 0

                    expanded_hypotheses = []

                    # create hypothesis
                    while len(expanded_hypotheses) < current_beam_size:

                        # Ignore UNK and EOS (for the first min_queries)
                        while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                                        arg_sort[
                                            arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                            arg_sort_i += 1

                        new_x = arg_sort[arg_sort_i]
                        arg_sort_i += 1

                        new_queries_accepted = queries_accepted + 1 if x == self.hred.eoq_symbol else queries_accepted
                        new_result = result + [new_x]
                        new_prob = softmax_out[0][new_x] * prob

                        expanded_hypotheses += [(new_prob, new_x, new_result, hidden_query, hidden_session,
                                                 hidden_decoder, new_queries_accepted)]

                    next_hypotheses += expanded_hypotheses

                # sort hypotheses and remove the least likely
                next_hypotheses = sorted(next_hypotheses, key=lambda x: x[0], reverse=True)[:current_beam_size]
                current_hypotheses = []

                for hypothesis in next_hypotheses:
                    _, x, _, _, _, _, queries_accepted = hypothesis

                    if x == self.hred.eos_symbol:
                        final_hypotheses += [hypothesis]
                        current_beam_size -= 1
                    else:
                        current_hypotheses += [hypothesis]

            final_hypotheses += current_hypotheses

            input_x = np.array(input_x).flatten()
            print('Sample input:  %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))

            for _, _, result, _, _, _, _ in final_hypotheses:
                result = np.array(result).flatten()
                print('Sample output: %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in result])))

            print('')

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
