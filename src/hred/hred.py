import tensorflow as tf
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

        self.vocab_size = 6
        self.embedding_size = 3
        self.query_hidden_size = 4
        self.session_hidden_size = 5
        self.decoder_hidden_size = self.query_hidden_size
        self.output_hidden_size = self.embedding_size
        self.eoq_symbol = 0  # End of Query symbol

    def step(self, X, start_hidden_query, start_hidden_session, start_hidden_decoder, start_output, start_logits):
        """
        Performs a step in the HRED

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

        # Making embeddings for x
        embedder = layers.embedding_layer(X, vocab_dim=self.vocab_size, embedding_dim=self.embedding_size)

        # Mask used to reset the query encoder when symbol is End-Of-Query symbol and to retain the state of the
        # session encoder when EoQ symbol has been seen yet.
        x_mask = tf.expand_dims(tf.cast(tf.not_equal(X, self.eoq_symbol), tf.float32), 2)

        # Computes the encoded query state. The tensorflow scan function repeatedly applies the gru_layer_with_reset
        # function to (embedder, x_mask) and it initialized the gru layer with the zero tensor.
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
            (embedder, x_mask),  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=start_hidden_query
        )

        query_encoder, _ = tf.unpack(query_encoder_packed, axis=1)

        # This part does the same, yet for the session encoder. Here we need to have the possibility to keep the current
        # state where we were at, namely if we have not seen a full query. If we have, update the session encoder state.
        session_encoder_packed = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_retain(
                result_prev[1], # h_retain_prev
                x,
                name='session_encoder',
                x_dim=self.query_hidden_size,
                y_dim=self.session_hidden_size
            ),
            (query_encoder, x_mask),
            initializer=start_hidden_session
        )

        session_encoder, _ = tf.unpack(session_encoder_packed, axis=1)

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
            (embedder, x_mask, session_encoder),  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=start_hidden_decoder
        )

        # After the decoder we add an additional output layer
        output = tf.scan(
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
        logits = tf.scan(
            lambda _, x: layers.logits_layer(
                x,
                x_dim=self.output_hidden_size,
                y_dim=self.vocab_size
            ),
            output,
            initializer=start_logits
        )

        return logits

    def softmax(self, logits):

        return tf.nn.softmax(logits)

    def loss(self, logits, labels):

        labels = tf.one_hot(labels, self.vocab_size)

        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        )

