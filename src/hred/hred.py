import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

import layers
import read_data


class HRED():

    def __init__(self):
        # We do not need to define parameters explicitly since tf.get_variable() also creates parameters for us

        self.vocab_size = 20
        self.embedding_size = 5
        self.query_hidden_size = 10
        self.session_hidden_size = 15
        self.decoder_hidden_size = self.query_hidden_size
        self.eoq_symbol = 0  # End of Query symbol

    def step(self, X):

        embedder = layers.embedding_layer(X, vocab_dim=self.vocab_size, embedding_dim=self.embedding_size)

        # Mask used to reset the query encoder when symbol is End-Of-Query symbol
        x_mask = tf.cast(tf.not_equal(X, self.eoq_symbol), tf.float32)

        query_encoder_packed = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_reset(
                result_prev[1],  # h_reset_prev
                x,
                name='query_encoder',
                x_dim=self.embedding_size,
                y_dim=self.query_hidden_size
            ),
            (embedder, x_mask),  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=tf.zeros((2, self.query_hidden_size))
        )

        query_encoder, _ = tf.unpack(query_encoder_packed, axis=1)

        session_encoder_packed = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_retain(
                result_prev[1], # h_retain_prev
                x,
                name='session_encoder',
                x_dim=self.query_hidden_size,
                y_dim=self.session_hidden_size
            ),
            (query_encoder, x_mask),
            initializer=tf.zeros((2, self.session_hidden_size))
        )

        session_encoder, _ = tf.unpack(session_encoder_packed, axis=1)

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
            initializer=tf.zeros((self.decoder_hidden_size, ))
        )

        output = layers.output_layer(
            decoder,
            embedder,
            x_dim=self.embedding_size,
            h_dim=self.decoder_hidden_size,
            y_dim=self.decoder_hidden_size
        )

        softmax = layers.softmax_layer(
            output,
            x_dim=self.decoder_hidden_size,
            y_dim=self.vocab_size
        )

        return softmax, output, decoder, session_encoder, query_encoder
