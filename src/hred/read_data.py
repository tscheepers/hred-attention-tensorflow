import tensorflow as tf
import os
import numpy as np

TRAIN_FILE = '../../data/tr_session.out'
VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
SMALL_FILE = '../../data/small_test_session.out'


def read_batch(data_file, batch_size=80, eoq_symbol=1, eos_symbol=2, pad_symbol=2, max_len=50):
    batch = ([], [])
    for i, (x, y) in enumerate(read_line(data_file, eoq_symbol, eos_symbol, pad_symbol, max_len)):

        if i != 0 and i % batch_size == 0:
            yield batch
            batch = ([], [])

        batch[0].append(x)
        batch[1].append(y)


def read_line(data_file, eoq_symbol=1, eos_symbol=2, pad_symbol=2, max_len=50):
    with open(data_file, 'r') as df:
        for line in df:

            # first replace tab with eoq symbol
            input = [int(x) for x in line.strip().replace('\t', ' %d ' % eoq_symbol).split()]
            label = input[1:] + [eos_symbol]

            # If the length of the current session is longer than max len, we remove the part that is too much
            if len(input) > max_len:
                input = input[:max_len]
                label = label[:max_len - 1] + [eos_symbol]
            else:
                padding = [pad_symbol for i in range(max_len - len(input))]
                input += padding
                label += padding

            yield input, label
