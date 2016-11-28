import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers

import layers
import read_data


class HRED():
    """ In this class we implement the steps that are needed in the HRED tf graph.
     The step function is called in train.py.

     The HRED consists of a query encoder. The query encodings are fed to a next encoder,
     the session encoder. This session encoder encodes all queries in the session.
     The output of the session encoder is fed to a decoder. The decoder outputs a query
     suggestion, i.e. it predicts the words that are likely to be entered by the user given
     the previous queries in the session.
    """

    def __init__(self):
        # We do not need to define parameters explicitly since tf.get_variable() also creates parameters for us

        self.vocab_size = 2504
        self.embedding_size = 64
        self.query_hidden_size = 128
        self.session_hidden_size = 256
        self.decoder_hidden_size = self.query_hidden_size
        self.output_hidden_size = self.embedding_size
        self.eoq_symbol = 1  # End of Query symbol
        self.eos_symbol = 2  # End of Session symbol

        self.start_hidden_query = tf.placeholder(tf.float32, (2, None, self.query_hidden_size))
        self.start_hidden_session = tf.placeholder(tf.float32, (2, None, self.session_hidden_size))
        self.start_hidden_decoder = tf.placeholder(tf.float32, (None, self.decoder_hidden_size))
        self.start_output = tf.placeholder(tf.float32, (None, self.output_hidden_size))
        self.start_logits = tf.placeholder(tf.float32, (None, self.vocab_size))

    def populate_feed_dict_with_defaults(self, batch_size=1, feed_dict=None):
        """
        Add zero hidden states to a feed dict
        """

        if feed_dict is None:
            feed_dict = dict()

        feed_dict[self.start_hidden_query] = np.zeros((2, batch_size, self.query_hidden_size))
        feed_dict[self.start_hidden_session] = np.zeros((2, batch_size, self.session_hidden_size))
        feed_dict[self.start_hidden_decoder] = np.zeros((batch_size, self.decoder_hidden_size))
        feed_dict[self.start_output] = np.zeros((batch_size, self.output_hidden_size))
        feed_dict[self.start_logits] = np.zeros((batch_size, self.vocab_size))

        return feed_dict

    def step_through_session(self, X, start_hidden_query=None, start_hidden_session=None, start_hidden_decoder=None,
                      start_output=None, start_logits=None, return_last_with_hidden_states=False, return_softmax=False):
        """
        Train for a batch of sessions in the HRED X can be a 3-D tensor (steps, batch, vocab)

        :param X: The input sessions. Lists of ints, ints correspond to words
        Shape: (max_length x batch_size)
        :param start_hidden_query: The first hidden state of the query encoder. Initialized with zeros.
        Shape: (2 x batch_size x query_hidden_size)
        :param start_hidden_session: The first hidden state of the session encoder. Iniitalized with zeros.
        Shape: (2 x batch_size x session_hidden_size)
        :param start_hidden_decoder: The first hidden state of the decoder. Initialized with zeros.
        Shape: (batch_size x output_hidden_size)
        :param start_output:
        :param start_logits:
        :return:
        """

        if start_hidden_query is None: start_hidden_query = self.start_hidden_query
        if start_hidden_session is None: start_hidden_session = self.start_hidden_session
        if start_hidden_decoder is None: start_hidden_decoder = self.start_hidden_decoder
        if start_output is None: start_output = self.start_output
        if start_logits is None: start_logits = self.start_logits

        # X = tf.Print(X, [X[:, 0]], message="This is X: ", summarize=20)

        # Making embeddings for x
        embedder = layers.embedding_layer(X, vocab_dim=self.vocab_size, embedding_dim=self.embedding_size)

        # Mask used to reset the query encoder when symbol is End-Of-Query symbol and to retain the state of the
        # session encoder when EoQ symbol has been seen yet.
        eoq_mask = tf.expand_dims(tf.cast(tf.not_equal(X, self.eoq_symbol), tf.float32), 2)
        # eoq_mask = tf.Print(eoq_mask, [eoq_mask[:,0,:]], message="This is eoq_mask: ", summarize=20)

        # Computes the encoded query state. The tensorflow scan function repeatedly applies the gru_layer_with_reset
        # function to (embedder, eoq_mask) and it initialized the gru layer with the zero tensor.
        # In the query encoder we need the possibility to reset the gru layer, namely after the eos symbol has been
        # reached
        query_encoder_packed = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_reset(
                result_prev[1],  # h_reset_prev
                x,
                name='query_encoder',
                x_dim=self.embedding_size,
                y_dim=self.query_hidden_size
            ),
            (embedder, eoq_mask),  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=start_hidden_query
        )

        query_encoder, hidden_query = tf.unpack(query_encoder_packed, axis=1)

        # This part does the same, yet for the session encoder. Here we need to have the possibility to keep the current
        # state where we were at, namely if we have not seen a full query. If we have, update the session encoder state.
        session_encoder_packed = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_retain(
                result_prev[1],  # h_retain_prev
                x,
                name='session_encoder',
                x_dim=self.query_hidden_size,
                y_dim=self.session_hidden_size
            ),
            (query_encoder, eoq_mask),
            initializer=start_hidden_session
        )

        session_encoder, hidden_session = tf.unpack(session_encoder_packed, axis=1)

        # This part makes the decoder for a step. The decoder uses both the word embeddings, the reset/retain vector
        # and the session encoder, so we give three variables to the decoder GRU. The decoder GRU is somewhat special,
        # as it incorporates the session_encoder into each hidden state update
        decoder = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_state_reset(
                result_prev,
                x,
                name='decoder',
                x_dim=self.embedding_size,
                h_dim=self.session_hidden_size,
                y_dim=self.decoder_hidden_size
            ),
            (embedder, eoq_mask, session_encoder),  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=start_hidden_decoder
        )

        # After the decoder we add an additional output layer
        # TODO: This should not have to be a scan function but  because of Tensorflow's 2-D matmul function we do this
        # for now. Perhaps with tf.batch_matmul ?
        output_layer = tf.scan(
            lambda _, x: layers.output_layer(
                x,
                x_dim=self.embedding_size,
                h_dim=self.decoder_hidden_size,
                s_dim=self.session_hidden_size,
                y_dim=self.output_hidden_size
            ),
            (decoder, embedder, session_encoder),
            initializer=start_output
        )

        # We compute the output logits based on the output layer above
        # TODO: This should not have to be a scan function but because of Tensorflow's 2-D matmul function we do this
        # for now. Perhaps with tf.batch_matmul ? tf.nn.softmax does accept a 3-D tensor however.
        logits = tf.scan(
            lambda _, x: layers.logits_layer(
                x,
                x_dim=self.output_hidden_size,
                y_dim=self.vocab_size
            ),
            output_layer,
            initializer=start_logits
        )

        # If we want the softmax back from this step or just the logits for the loss function
        if return_softmax:
            output = self.softmax(logits)
        else:
            output = logits

        # If we want to continue decoding with single_step we need the hidden states of all GRU layers
        if return_last_with_hidden_states:
            # hidden_decoder = decoder  # there is no resetted decoder output
            # return output[-1, :, :], hidden_query[-1, :, :], hidden_session[-1, :, :], hidden_decoder[-1, :, :]
            return None
        else:
            return output

    def single_step(self, X, prev_hidden_query, prev_hidden_session, prev_hidden_decoder):
        """
        Performs a step in the HRED X can be a 2-D tensor (batch, vocab), this can be used
        for beam search

        :param X: The input sessions. Lists of ints, ints correspond to words
        Shape: (max_length)
        :param start_hidden_query: The first hidden state of the query encoder. Initialized with zeros.
        Shape: (2 x query_hidden_size)
        :param start_hidden_session: The first hidden state of the session encoder. Iniitalized with zeros.
        Shape: (2 x session_hidden_size)
        :param start_hidden_decoder: The first hidden state of the decoder. Initialized with zeros.
        Shape: (output_hidden_size)
        :return:
        """

        # Making embeddings for x
        embedder = layers.embedding_layer(X, vocab_dim=self.vocab_size, embedding_dim=self.embedding_size, reuse=True)

        # Mask used to reset the query encoder when symbol is End-Of-Query symbol and to retain the state of the
        # session encoder when EoQ symbol has been seen yet.
        eoq_mask = tf.cast(tf.not_equal(X, self.eoq_symbol), tf.float32)

        query_encoder, hidden_query = tf.unpack(layers.gru_layer_with_reset(
            prev_hidden_query,  # h_reset_prev
            (embedder, eoq_mask),
            name='query_encoder',
            x_dim=self.embedding_size,
            y_dim=self.query_hidden_size,
            reuse=True
        ))

        # This part does the same, yet for the session encoder. Here we need to have the possibility to keep the current
        # state where we were at, namely if we have not seen a full query. If we have, update the session encoder state.
        session_encoder, hidden_session = tf.unpack(layers.gru_layer_with_retain(
            prev_hidden_session,  # h_retain_prev
            (query_encoder, eoq_mask),
            name='session_encoder',
            x_dim=self.query_hidden_size,
            y_dim=self.session_hidden_size,
            reuse=True
        ))

        # This part makes the decoder for a step. The decoder uses both the word embeddings, the reset/retain vector
        # and the session encoder, so we give three variables to the decoder GRU. The decoder GRU is somewhat special,
        # as it incorporates the session_encoder into each hidden state update
        hidden_decoder = layers.gru_layer_with_state_reset(
            prev_hidden_decoder,
            (embedder, eoq_mask, session_encoder),
            name='decoder',
            x_dim=self.embedding_size,
            h_dim=self.session_hidden_size,
            y_dim=self.decoder_hidden_size,
            reuse=True
        )

        decoder = hidden_decoder

        # After the decoder we add an additional output layer
        output = layers.output_layer(
            (decoder, embedder, session_encoder),
            x_dim=self.embedding_size,
            h_dim=self.decoder_hidden_size,
            s_dim=self.session_hidden_size,
            y_dim=self.output_hidden_size,
            reuse=True
        )

        # We compute the output logits based on the output layer above
        logits = layers.logits_layer(
            output,
            x_dim=self.output_hidden_size,
            y_dim=self.vocab_size,
            reuse=True
        )

        softmax = self.softmax(logits)

        return softmax, hidden_query, hidden_session, hidden_decoder

    def softmax(self, logits):
        """
        Perform a simple softmax function on the logits
        logits can be a 3-D tensor or a 2-D tensor
        """

        return tf.nn.softmax(logits)

    def loss(self, X, logits, labels):
        """
        Calculate the loss for logits. both logits and
        labels can be both a 3-D and 2-D tensor

        You do not have to pass a one hot vector for the labels,
        this does this method for you
        """

        # labels = tf.Print(labels, [labels[:, 1]], message="This is labels: ", summarize=5)

        labels = tf.one_hot(labels, self.vocab_size)

        # logits = tf.Print(logits, [tf.reduce_max(logits)], message="This is max logits: ")
        # logits = tf.Print(logits, [tf.reduce_min(logits)], message="This is min logits: ")
        # logits = tf.Print(logits, [tf.reduce_sum(logits, reduction_indices=[2])[:, 1]], message="This is sum logits: ", summarize=5)

        eos_mask = tf.expand_dims(tf.cast(tf.not_equal(X, self.eos_symbol), tf.float32), 2)
        labels = labels * eos_mask

        # loss = -tf.reduce_sum(labels * tf.log(logits))
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        )

        return loss
