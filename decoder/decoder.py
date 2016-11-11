import random
import numpy as np
import ptb_reader
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

# load up our text
from os import path

train_data, valid_data, test_data, vocabulary_size = ptb_reader.ptb_raw_data(path.join(path.dirname(__file__), 'data'))

model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=64))
model.add(LSTM(64))
model.add(Dense(vocabulary_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')