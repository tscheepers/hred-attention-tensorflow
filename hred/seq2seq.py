from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn import seq2seq

class Seq2SeqModel(object):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

        # SAMPLED LOSS HERE? --> SEE CODE
        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)

            return tf.cast(
                tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                           num_samples, self.target_vocab_size),
                dtype)

        softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        # if use_lstm:
        #     single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        # if num_layers > 1:
        #     cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # This is making the word embeddings
        def seq2seq_f(encoder_inputs, decoder_inputs,
                      num_encoder_symbols=source_vocab_size,
                      num_decoder_symbols=target_vocab_size,
                      output_projection=output_projection,
                      do_decode=False):
            outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols, num_decoder_symbols,
                output_projection=output_projection, feed_previous=do_decode)
            return outputs, states

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        # TODO: Not using BUCKETS at this time
        for i in range(10):
            self.encoder_inputs.append(tf.placeholder(tf.int32,
                                                      shape=[None],
                                                      name="encoder{0}".format(i)))
            self.decoder_inputs.append(tf.placeholder(tf.int32,
                                                      shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.int32,
                                                      shape=[None],
                                                      name="encoder{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses #TODO This is a lot different from the sample code
        if not forward_only:
            self.outputs, self.states = lambda x, y: seq2seq_f(x, y)
        else:
            self.outputs, self.states = lambda x, y: seq2seq_f(x, y, do_decode=True)

        # Gradients and SGD update operation for training the model
        params = tf.trainable_variables()
