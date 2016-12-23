""" File to build to perform beamsearch for the next query
"""

import numpy as np
import tensorflow as tf
import cPickle
import sordoni.data_iterator as sordoni_data_iterator
from utils import make_attention_mask

from hred import HRED

tf.logging.set_verbosity(tf.logging.DEBUG)

VALIDATION_FILE = '../../data/val_ngram_session.out'
TEST_FILE = '../../data/test_ngram_session.out'
LOGS_DIR = '../../logs'
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2
RESTORE = True

N_BUCKETS = 20
MAX_ITTER = 10000000

CHECKPOINT_FILE = '../../checkpoints/model_ngram-huge.ckpt'
# OUR_VOCAB_FILE = '../../data/aol_vocab_50000.pkl'
# OUR_TRAIN_FILE = '../../data/aol_sess_50000.out'
# OUR_SAMPLE_FILE = '../../data/sample_aol_sess_50000.out'
SORDONI_VOCAB_FILE = '../../data/sordoni/ngram/all/train.dict.pkl'
SORDONI_TRAIN_FILE = '../../data/sordoni/ngram/all/train.ses.pkl'
SORDONI_VALID_FILE = '../../data/sordoni/ngram/all/valid.ses.pkl'
VOCAB_SIZE = 50003
# EMBEDDING_DIM = 25
# QUERY_DIM = 50
# SESSION_DIM = 100
EMBEDDING_DIM = 128
QUERY_DIM = 512
SESSION_DIM = 1024
BATCH_SIZE = 1
MAX_LENGTH = 256

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


# For beam search example:
# https://github.com/tensorflow/tensorflow/issues/654#issuecomment-168237741

class Sampler(object):
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
        self.attention_mask = tf.placeholder(tf.float32, shape=(max_length, batch_size, max_length))

        self.X_sample = tf.placeholder(tf.int64, shape=(batch_size,))
        self.H_query = tf.placeholder(tf.float32, shape=(None, batch_size, self.hred.query_dim))
        self.H_session = tf.placeholder(tf.float32, shape=(batch_size, self.hred.session_dim))
        self.H_decoder = tf.placeholder(tf.float32, shape=(batch_size, self.hred.decoder_dim))

        self.logits = self.hred.step_through_session(self.X, self.attention_mask)
        self.loss = self.hred.loss(self.X, self.logits, self.Y)
        self.softmax = self.hred.softmax(self.logits)
        self.accuracy = self.hred.non_padding_accuracy(self.logits, self.Y)
        self.non_symbol_accuracy = self.hred.non_symbol_accuracy(self.logits, self.Y)

        self.session_inference = self.hred.step_through_session(
             self.X, self.attention_mask, return_softmax=True, return_last_with_hidden_states=True, reuse=True
        )
        self.step_inference = self.hred.single_step(
             self.X_sample, self.H_query, self.H_session, self.H_decoder, reuse=True
        )

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def beam_sample(self, input_x, max_steps=30, beam_size=25, min_queries=1):

        attention_mask = make_attention_mask(input_x)

        softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
            self.session_inference,
            feed_dict={self.X: input_x, self.attention_mask: attention_mask}
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

                input_for_mask = np.concatenate((input_x, np.expand_dims(np.array(result), axis=1)), axis=0)
                attention_mask = make_attention_mask(input_for_mask)

                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                    self.step_inference,
                    {self.X_sample: [x], self.H_query: hidden_query, self.H_session: hidden_session,
                     self.H_decoder: hidden_decoder, self.attention_mask: attention_mask}
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
                                        arg_sort_i] == self.hred.eos_symbol and queries_accepted <= min_queries):
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

                if x == self.hred.eoq_symbol:
                    final_hypotheses += [hypothesis]
                    current_beam_size -= 1
                else:
                    current_hypotheses += [hypothesis]

        final_hypotheses += current_hypotheses

        input_x = np.array(input_x).flatten()
        print('Sample input:  %s' % (''.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))

        print('Sample output:')

        for _, _, result, _, _, _, _ in final_hypotheses:
            result = np.array(result).flatten()
            print('%s' % (''.join([self.vocab_lookup_dict.get(x, '?') for x in result])))

        print('')

        return final_hypotheses


def get_batch(train_data):
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

        sampler = Sampler()

        with tf.Session() as sess:

            # Restore variables from disk.
            sampler.saver.restore(sess, CHECKPOINT_FILE)
            print("Model restored.")

            for i in range(250):

                x_batch, _, seq_len = get_batch(sampler.valid_data)
                input_x = np.expand_dims(x_batch[:, 0], axis=1)

                sampler.beam_sample(input_x)



