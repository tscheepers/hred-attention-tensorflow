import random
import numpy as np
import ptb_reader
from keras.layers import Input, Embedding, TimeDistributedDense, TimeDistributed
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
import keras.backend as K
import tensorflow as tf

# load up our text
from os import path

train_raw, valid_raw, test_raw, vocab_size = ptb_reader.ptb_raw_data(path.join(path.dirname(__file__), 'data'))

window = 10


def data(raw):
    n_train = len(raw) - window

    X = np.zeros((n_train, window), dtype=int)
    Y = np.zeros((n_train), dtype=int)

    for i in range(n_train):
        for j in range(window):
            X[i, j] = train_raw[i + j]
        Y[i] = train_raw[i + window]

    return X, Y


def one_hot_categorical_crossentropy(y_true, y_pred):
    y_true1 = tf.one_hot(K.flatten(K.cast(y_true, 'int32')), vocab_size, axis=1)
    return K.categorical_crossentropy(y_true1, y_pred)


model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=window))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss=one_hot_categorical_crossentropy, optimizer='rmsprop')

X_train, Y_train = data(train_raw)
X_val, Y_val = data(valid_raw)
# X_test, Y_test = data(valid_raw)

model.fit(X_train, Y_train, batch_size=128, nb_epoch=10, validation_data=(X_val, Y_val))
