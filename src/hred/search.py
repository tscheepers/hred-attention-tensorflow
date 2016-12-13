""" File to build to perform beamsearch for the next query
"""

import numpy as np
import tensorflow as tf
import cPickle
import sordoni.data_iterator as sordoni_data_iterator

from hred import HRED

tf.logging.set_verbosity(tf.logging.DEBUG)


VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
LOGS_DIR = '../../logs'
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2

N_BUCKETS = 20

CHECKPOINT_FILE = '../../checkpoints/model-large.ckpt'
# OUR_VOCAB_FILE = '../../data/aol_vocab_50000.pkl'
# OUR_TRAIN_FILE = '../../data/aol_sess_50000.out'
# OUR_SAMPLE_FILE = '../../data/sample_aol_sess_50000.out'
SORDONI_VOCAB_FILE = '../../data/sordoni/all/train.dict.pkl'
SORDONI_TRAIN_FILE = '../../data/sordoni/all/train.ses.pkl'
SORDONI_VALID_FILE = '../../data/sordoni/all/valid.ses.pkl'
VOCAB_SIZE = 50003
EMBEDDING_DIM = 25
QUERY_DIM = 50
SESSION_DIM = 100
BATCH_SIZE = 80
MAX_LENGTH = 50

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

if __name__ == '__main__':
    with tf.Graph().as_default():

        vocab = cPickle.load(open(SORDONI_VOCAB_FILE, 'r'))
        vocab_lookup_dict = {k: v for v, k, count in vocab}

        train_data, valid_data = sordoni_data_iterator.get_batch_iterator(np.random.RandomState(SEED), {
            'eoq_sym': EOQ_SYMBOL,
            'eos_sym': EOS_SYMBOL,
            'sort_k_batches': N_BUCKETS,
            'bs': BATCH_SIZE,
            'train_session': SORDONI_TRAIN_FILE,
            'seqlen': MAX_LENGTH,
            'valid_session': SORDONI_VALID_FILE
        })
        train_data.start()
        valid_data.start()

        vocab_size = len(vocab_lookup_dict)

        # vocab_size = VOCAB_SIZE
        # vocab_lookup_dict = read_data.read_vocab_lookup(OUR_VOCAB_FILE)

        hred = HRED(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, query_dim=QUERY_DIM,
                         session_dim=SESSION_DIM, decoder_dim=QUERY_DIM, output_dim=EMBEDDING_DIM,
                         eoq_symbol=EOQ_SYMBOL, eos_symbol=EOS_SYMBOL, unk_symbol=UNK_SYMBOL)

        batch_size = None
        max_length = None

        X = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        X_beam = tf.placeholder(tf.int64, shape=(batch_size, ))
        H_query = tf.placeholder(tf.float32, shape=(batch_size, hred.query_dim))
        H_session = tf.placeholder(tf.float32, shape=(batch_size, hred.session_dim))
        H_decoder = tf.placeholder(tf.float32, shape=(batch_size, hred.decoder_dim))

        session_result = hred.step_through_session(X, return_last_with_hidden_states=True, return_softmax=True)
        step_result = hred.single_step(X_beam, H_query, H_session, H_decoder)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:

            # Restore variables from disk.
            saver.restore(sess, CHECKPOINT_FILE)
            print("Model restored.")

            x = np.array([10, 11])
            x = np.expand_dims(x, 1)
            batch_size = 1

            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(session_result, {X: x})

            arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]
            # Ignore UNK and EOS (for the first min_queries)
            arg_sort_i = 0
            while arg_sort[arg_sort_i] == hred.unk_symbol or (
                            arg_sort[arg_sort_i] == hred.eos_symbol and queries_accepted < min_queries):
                arg_sort_i += 1
            x = arg_sort[arg_sort_i]
            print(vocab_lookup_dict.get(x, '?'))

            i = 0
            max_i = 100

            while x != hred.eos_symbol and i < max_i:

                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                    step_result,
                    {X_beam: [x], H_query: hidden_query, H_session: hidden_session, H_decoder: hidden_decoder}
                )

                arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]
                # Ignore UNK and EOS (for the first min_queries)
                arg_sort_i = 0
                while arg_sort[arg_sort_i] == hred.unk_symbol or (
                                arg_sort[arg_sort_i] == hred.eos_symbol and queries_accepted < min_queries):
                    arg_sort_i += 1
                x = arg_sort[arg_sort_i]

                if x == hred.eoq_symbol:
                    queries_accepted += 1

                print(vocab_lookup_dict.get(x, '?'))

                i += 1