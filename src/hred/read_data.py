import tensorflow as tf
import os
import numpy as np

TRAIN_FILE = '../../data/tr_session.out'
VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
SMALL_FILE = '../../data/small_test_session.out'


def read_batch(data_file, batch_size=80, eoq_symbol=1, pad_symbol=2, max_seq_len=50):
    batch = ([], [])
    for i, (x, y) in enumerate(read_line(data_file, eoq_symbol)):

        if i != 0 and i % batch_size == 0:
            padded_batch, max_len = add_padding(batch, eoq_symbol, pad_symbol, max_seq_len)
            yield padded_batch, max_len
            batch = ([], [])

        batch[0].append(x)
        batch[1].append(y)


def read_line(data_file, eoq_symbol=1):
    with open(data_file, 'r') as df:
        for line in df:

            # first replace tab with eoq symbol, never predict eos_symbol
            x = [int(i) for i in line.strip().replace('\t', ' %d ' % eoq_symbol).split()]
            y = x[1:] + [eoq_symbol]

            # input = [int(x) for x in line.strip().replace('\t', ' %d ' % eoq_symbol).split()] + [eoq_symbol]
            # label = input[1:] + [eos_symbol]
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
