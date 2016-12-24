""" File to build to perform beamsearch for the next query
"""

import numpy as np
import tensorflow as tf
import cPickle
import sordoni.data_iterator as sordoni_data_iterator
from utils import make_attention_mask

from hred import HRED

tf.logging.set_verbosity(tf.logging.DEBUG)

CONTEXT_FILE = '../../data/2_exp-context-length/test_c1_hred_sess.ctx'
CANDIDATE_FILE = '../../data/2_exp-context-length/test_c1_hred_cand.ctx'
OUTPUT_SCORE_FILE = '../../data/2_exp-context-length/test_c1_score'

FILE_PREFIXES = [
    '../../data/1_exp-baseline/test_all_hred',
    '../../data/1_exp-baseline/val_all_hred',
    '../../data/1_exp-baseline/tr_all_hred',
    '../../data/2_exp-context-length/test_c1_hred',
    '../../data/2_exp-context-length/val_c1_hred',
    '../../data/2_exp-context-length/tr_c1_hred',
    '../../data/2_exp-context-length/test_c2_hred',
    '../../data/2_exp-context-length/val_c2_hred',
    '../../data/2_exp-context-length/tr_c2_hred',
    '../../data/2_exp-context-length/test_c3_hred',
    '../../data/2_exp-context-length/val_c3_hred',
    '../../data/2_exp-context-length/tr_c3_hred',
    '../../data/3_exp-noisy/test_noisy_hred',
    '../../data/3_exp-noisy/val_noisy_hred',
    '../../data/3_exp-noisy/tr_noisy_hred',
    '../../data/4_exp-long-tail/test_long_all_hred',
    '../../data/4_exp-long-tail/val_long_all_hred',
    '../../data/4_exp-long-tail/tr_long_all_hred',
]

VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
LOGS_DIR = '../../logs'
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2
RESTORE = False

N_BUCKETS = 20
MAX_ITTER = 10000000

CHECKPOINT_FILE = '../../checkpoints/model-huge-attention.ckpt'
# OUR_VOCAB_FILE = '../../data/aol_vocab_50000.pkl'
# OUR_TRAIN_FILE = '../../data/aol_sess_50000.out'
# OUR_SAMPLE_FILE = '../../data/sample_aol_sess_50000.out'
SORDONI_VOCAB_FILE = '../../data/sordoni/all/train.dict.pkl'
SORDONI_TRAIN_FILE = '../../data/sordoni/all/train.ses.pkl'
SORDONI_VALID_FILE = '../../data/sordoni/all/valid.ses.pkl'
VOCAB_SIZE = 50003
# EMBEDDING_DIM = 25
# QUERY_DIM = 50
# SESSION_DIM = 100
EMBEDDING_DIM = 64
QUERY_DIM = 128
SESSION_DIM = 256
BATCH_SIZE = 80
MAX_LENGTH = 50

SEED = 1234

# For beam search example:
# https://github.com/tensorflow/tensorflow/issues/654#issuecomment-168237741

class Scorer(object):
    def __init__(self):

        vocab = cPickle.load(open(SORDONI_VOCAB_FILE, 'r'))
        self.vocab_lookup_dict = {k: v for v, k, count in vocab}
        self.vocab_dict = {k: v for k, v, count in vocab}

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

    def score_candidates(self, context, candidates, output_file):

        # print("Start scoring.")

        input_for_mask = np.expand_dims(np.array(context), axis=1)
        attention_mask = make_attention_mask(input_for_mask)

        # print(input_for_mask)
        # print(attention_mask)

        context_softmax_out, context_hidden_query, context_hidden_session, context_hidden_decoder = sess.run(
            self.session_inference,
            feed_dict={
                self.X: input_for_mask,
                self.attention_mask: attention_mask
            }
        )

        scores = []

        for candidate in candidates:

            probs = []

            hidden_query = context_hidden_query
            hidden_session = context_hidden_session
            hidden_decoder = context_hidden_decoder
            softmax_out = context_softmax_out[0]

            input_uptil_now = context

            for token in candidate:
                if softmax_out.shape[0] == 1:
                    softmax_out = softmax_out[0]

                probs += [-np.log2(softmax_out[token])]
                input_uptil_now += [token]

                input_for_mask = np.expand_dims(np.array(input_uptil_now), axis=1)
                attention_mask = make_attention_mask(input_for_mask)

                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                    self.step_inference,
                    {self.X_sample: [token], self.H_query: hidden_query, self.H_session: hidden_session,
                     self.H_decoder: hidden_decoder,
                     self.attention_mask: attention_mask
                     }
                )

            score = np.mean(np.array(probs))
            # print("score: %f" % score)

            output_file.write('%f\n' % score)
            scores += [score]

        return scores


if __name__ == '__main__':
    with tf.Graph().as_default():
        scorer = Scorer()

        with tf.Session() as sess:
            # Restore variables from disk.
            scorer.saver.restore(sess, CHECKPOINT_FILE)
            print("Model restored.")

            for prefix in FILE_PREFIXES:

                context_file = prefix + '_sess.ctx'
                candidate_file = prefix + '_cand.ctx'
                output_source_file = prefix + '_score_attention'

                all_contexts = [[scorer.vocab_dict.get(z, EOQ_SYMBOL) for z in y.split()] for y in open(context_file).readlines()]

                print("Context loaded: %s" % context_file)

                all_candidates = [[[scorer.vocab_dict.get(z, EOQ_SYMBOL) for z in y.split()] for y in x.split('\t')] for x in
                              open(candidate_file).readlines()]

                print("Candidates loaded: %s" % candidate_file)

                scores = []
                i = 0

                with open(output_source_file, 'w') as output_file:

                    for context, candidates in zip(all_contexts, all_candidates):

                        if i % 100 == 0:
                            print('%f %%' % (100*i/float(len(all_contexts))))

                        scores += scorer.score_candidates(context + [EOQ_SYMBOL], candidates, output_file)

                        i += 1

                print("Scores saved: %s" % output_source_file)
