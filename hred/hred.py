from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentContainer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input


def SimpleED(input_length, output_length, hidden_dim, input_dim, batch_size, dropout):
    """
    See: https://github.com/farizrahman4u/seq2seq/blob/master/seq2seq/models.py

    :param input_length:
    :param output_length:
    :param hidden_dim:
    :param input_dim:
    :param batch_size:
    :param dropout:
    :return:
    """

    encoder = RecurrentContainer(input_length=input_length)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(batch_size, input_dim)))
    # encoder.add(Dropout(dropout))
    encoder.add(LSTMCell(hidden_dim))

    decoder = RecurrentContainer(decode=True, output_length=output_length,
                                 input_length=input_length)
    decoder.add(Dropout(dropout, batch_input_shape=(batch_size, hidden_dim)))
    decoder.add(LSTMCell(hidden_dim))
    # decoder.add(Dropout(dropout))
    decoder.add(LSTMCell(hidden_dim))

    model = Sequential()
    model.add(encoder)
    model.add(decoder)

    return model
