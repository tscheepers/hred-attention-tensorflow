from __future__ import absolute_import

import keras
import numpy as np
from recurrentshop import GRUCell, RecurrentContainer
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
    encoder.add(GRUCell(hidden_dim, batch_input_shape=(batch_size, input_dim)))
    # encoder.add(Dropout(dropout))
    # encoder.add(GRUCell(hidden_dim))

    decoder = RecurrentContainer(decode=True, output_length=output_length,
                                 input_length=input_length)
    decoder.add(Dropout(dropout, batch_input_shape=(batch_size, hidden_dim)))
    decoder.add(GRUCell(hidden_dim))
    # decoder.add(Dropout(dropout))
    # decoder.add(GRUCell(hidden_dim))

    model = Sequential()
    model.add(encoder)
    model.add(decoder)

    return model

if __name__ == '__main__':
    max_vocab = 10

    X_train = [np.eye(max_vocab)[x] for x in np.array([[3,1,4,9], [5,7,9], [2,4,6,1,3,9]])]
    Y_train = [np.eye(max_vocab)[x] for x in np.array([[5,6,7,8,9], [1,3,9], [8,2,9]])]

    batch_size = 1
    input_length = max_vocab
    input_dim = 3
    hidden_dim = 3
    output_length = max_vocab

    model = SimpleED(input_length, output_length, hidden_dim, input_dim, batch_size, 0)

    model.summary()

    model.compile(loss='mse', optimizer='rmsprop')

    model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=100,
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)],
          verbose=1)

    print("model")
