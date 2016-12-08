import cPickle as pickle
import os
import logging as logger

VOCAB_FILE = '../../data/aol_vocab_50000.pkl'


def load_vocab(vocab_file):
    assert os.path.isfile(vocab_file)
    vocab = dict([(x[0], x[1]) for x in pickle.load(open(vocab_file, "r"))])
    # Check consistency
    assert '<unk>' in vocab
    assert '</s>' in vocab
    assert '</q>' in vocab
    assert '</p>' in vocab
    logger.info("INFO - Successfully loaded vocabulary dictionary %s." % vocab_file)
    logger.info("INFO - Vocabulary contains %d words" % len(vocab))
    return vocab

v = load_vocab(VOCAB_FILE)
result = [4322, 4322, 4322, 4322, 4322, 4322, 4322]
result_words = [v.get(x) for x in result]

print result_words
print v
