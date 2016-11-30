import tensorflow as tf
import os
import numpy as np
import subprocess
import pickle
import os
import logging as logger


def read_batch(data_file, batch_size=80, eoq_symbol=1, pad_symbol=2, max_seq_len=50):
    batch = ([], [])
    subprocess.call('shuf %s -o %s' % (data_file, data_file), shell=True) # shuffling the file for the batches
    for i, (x, y) in enumerate(read_line(data_file, eoq_symbol)):
        if i != 0 and i % batch_size == 0:
            padded_batch, max_len = add_padding(batch, eoq_symbol, pad_symbol, max_seq_len)
            yield padded_batch, max_len
            batch = ([], [])

        batch[0].append(x)
        batch[1].append(y)


def read_line(data_file, eoq_symbol=1, eos_symbol=2, sos_symbol=3):
    with open(data_file, 'r') as df:
        for line in df:

            # first replace tab with eoq symbol, never predict eos_symbol
            # x = [int(i) for i in line.strip().replace('\t', ' %d ' % eoq_symbol).split()]
            # y = x[1:] + [eoq_symbol]

            x = [sos_symbol] + [int(i) for i in line.strip().replace('\t', ' %d ' % eoq_symbol).split()] + [eoq_symbol]
            y = x[1:] + [eos_symbol]

            yield x, y


def add_padding(batch, eoq_symbol=1, pad_symbol=2, max_seq_len=50):

    max_len_x = len(max(batch[0], key=len))
    max_len_y = len(max(batch[1], key=len))
    max_len = min(max(max_len_x, max_len_y), max_seq_len)
    padded_batch = ([], [])

    # If the length of the current session is longer than max len, we remove the part that is too much
    for i in range(len(batch[0])):
        x = batch[0][i]
        y = batch[1][i]
        if len(x) > max_len:
            x = x[:max_len]
            y = y[:max_len - 1] + [eoq_symbol]
        else:
            padding = [pad_symbol for j in range(max_len - len(x))]
            x += padding
            y += padding

        padded_batch[0].append(x)
        padded_batch[1].append(y)

    # Return max_len to keep track of this, to be able to adapt model
    return padded_batch, max_len


def add_padding_and_sort(batch, eoq_symbol, pad_symbol, max_seq_len):
    sorted_batch = batch.sort(key=len)
    add_padding(sorted_batch, eoq_symbol, pad_symbol, max_seq_len)


def read_vocab_lookup(vocab_file):
    vocab_shifted = read_token_lookup(vocab_file)
    return dict((v, k) for k, v in vocab_shifted.iteritems())


def read_token_lookup(vocab_file):
    assert os.path.isfile(vocab_file)
    vocab = pickle.load(open(vocab_file, "r"))
    # vocab = dict([(x[0], x[1]) for x in loaded_file])

    # Check consistency
    if '<unk>' not in vocab:
        vocab['<unk>'] = 0
    if '</q>' not in vocab:
        vocab['</q>'] = 1
    if '</s>' not in vocab:
        vocab['</s>'] = 2
    if '</p>' not in vocab:
        vocab['</p>'] = 3

    logger.info("INFO - Successfully loaded vocabulary dictionary %s." % vocab_file)
    logger.info("INFO - Vocabulary contains %d words" % len(vocab))
    return vocab