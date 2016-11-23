import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from hred import layers


class Encoder():

    def __init__(self):
        # We do not need to define parameters explicitly since tf.get_variable() also creates parameters for us

        self.vocab_size = 10000
        self.embedding_size = 256
        self.query_hidden_size = 512
        self.session_hidden_size = 512
        self.eoq_symol = 1 # End of Query sumbol

        pass

    def step(self, X):

        embedder = layers.embedding_layer(X, vocab_dim=self.vocab_size, embedding_dim=self.embedding_size)

        # Mask used to reset the query encoder when symbol is End-Of-Query symbol
        x_mask = tf.not_equal(X, self.eoq_symol)

        query_encoder, _ = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_reset(
                result_prev[1],  # h_reset_prev
                x,
                name='query_encoder',
                x_dim=self.embedding_size,
                h_dim=self.query_hidden_size
            ),
            tf.pack([x_mask, embedder], axis=1),  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=tf.zeros((self.query_hidden_size, ))
        )

        session_encoder = tf.scan(
            lambda result_prev, x: layers.gru_layer(
                result_prev,
                x,
                name='session_encoder',
                x_dim=self.query_hidden_size,
                h_dim=self.session_hidden_size
            ),
            query_encoder,  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=tf.zeros((self.session_hidden_size, ))
        )

        decoder, _ = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_reset(
                result_prev[1],  # h_reset_prev
                x,
                name='decoder',
                x_dim=self.embedding_size,
                h_dim=self.query_hidden_size
            ),
            tf.pack([x_mask, embedder], axis=1),  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=tf.zeros((self.query_hidden_size, ))
        )

        return query_encoder, session_encoder