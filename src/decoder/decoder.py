# coding: utf-8
import os
import random
import numpy as np
import keras.backend as K
import tensorflow as tf
from nltk import word_tokenize

import ptb
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


class Decoder:

    def __init__(self):

        # The unfolding window for the RNN (during backprop)
        self.window = 10

    def load_data(self, ptb_path=None):
        """
        Load the Penn Tree Bank training data
        """

        train_raw, valid_raw, test_raw, vocab = ptb.ptb_raw_data(ptb_path)

        self.vocab = vocab
        self.train_data = self.process_raw_data(train_raw)
        self.val_data = self.process_raw_data(valid_raw)
        self.test_data = self.process_raw_data(test_raw)

        self.vocab_reversed = dict()

        for key, value in vocab.items():
            self.vocab_reversed[value] = key

    def process_raw_data(self, raw):
        """
        Transform the raw Penn Tree Bank data into a format the
        model accepts
        """

        n_train = len(raw) - self.window

        X = np.zeros((n_train, self.window), dtype=int)
        Y = np.zeros((n_train), dtype=int)

        for i in range(n_train):
            for j in range(self.window):
                X[i, j] = raw[i + j] + 1 # plus padding token
            Y[i] = raw[i + self.window] + 1 # plus padding token

        return X, Y

    def build_model(self):
        """
        Build the model
        """

        vocabsize = len(self.vocab) + 1 # plus padding token

        model = Sequential()
        model.add(Embedding(vocabsize, 512, input_length=self.window))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(512, return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(vocabsize))
        model.add(Activation('softmax'))

        def one_hot_categorical_crossentropy(y_true, y_pred):
            y_true1 = tf.one_hot(K.flatten(K.cast(y_true, 'int32')), vocabsize, axis=1)
            return K.categorical_crossentropy(y_true1, y_pred)

        model.compile(loss=one_hot_categorical_crossentropy, optimizer='rmsprop')

        self.model = model

    def load_weights(self, checkpoint_path=None):
        """
        Load the weights from a checkpoint
        """
        self.model.load_weights(checkpoint_path)

    def train(self, checkpoint_path=None):
        """
        Trains the model, before executing build the model and load the train data
        """
        checkpointer = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True)
        X_train, Y_train = self.train_data

        self.model.fit(
            X_train, Y_train,
            batch_size=128,
            nb_epoch=10,
            validation_data=self.val_data,
            callbacks=[checkpointer]
        )

    def predict(self, string):

        tokens = word_tokenize(string)
        sequence = [self.vocab.get(x, self.vocab.get('<unk>')) for x in tokens]
        X = pad_sequences([sequence], maxlen=self.window, dtype='int32')

        pad = self.vocab_reversed[0]
        unk = self.vocab_reversed[1]

        Y = self.model.predict(X)
        argmax = np.argmax(Y)
        word = self.vocab_reversed[argmax]
        return word


if __name__ == '__main__':

    decoder = Decoder()

    ptb_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ptb')
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'decoder')

    decoder.load_data(ptb_path)
    decoder.build_model()
    # decoder.load_weights(checkpoint_path)
    decoder.train(checkpoint_path)

    result = decoder.predict('hey how are you doing this time of')
    print(result)




