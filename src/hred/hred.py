import tensorflow as tf
import numpy as np

import layers


class HRED():
    """ In this class we implement the steps that are needed in the HRED tf graph.
     The step function is called in train.py.

     The HRED consists of a query encoder. The query encodings are fed to a next encoder,
     the session encoder. This session encoder encodes all queries in the session.
     The output of the session encoder is fed to a decoder. The decoder outputs a query
     suggestion, i.e. it predicts the words that are likely to be entered by the user given
     the previous queries in the session.
    """

    def __init__(self, vocab_size=50004, embedding_dim=300, query_dim=1000, session_dim=1500,
                 decoder_dim=1000, output_dim=50004, unk_symbol=0, eoq_symbol=1, eos_symbol=2):
        # We do not need to define parameters explicitly since tf.get_variable() also creates parameters for us

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.query_dim = query_dim
        self.session_dim = session_dim
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim
        self.unk_symbol = unk_symbol  # Unknown symbol
        self.eoq_symbol = eoq_symbol  # End of Query symbol
        self.eos_symbol = eos_symbol  # End of Session symbol

    # def seen_eoq(self):
    #     return tf.ones(shape=[1, 5])
    #
    # def not_seen_eoq(self):
    #     return tf.zeros(shape=[1, 5])

    def make_attention_mask(self, eoq=False):
        mask = [1, 2, 3, 4]
        return tf.Variable(mask)

    def step_through_session(self, X, return_last_with_hidden_states=False, return_softmax=False, reuse=False):
        """
        Train for a batch of sessions in the HRED X can be a 3-D tensor (steps, batch, vocab)

        :param X: The input sessions. Lists of ints, ints correspond to words
        Shape: (max_length x batch_size)
        :return:
        """

        num_of_steps = tf.shape(X)[0]
        batch_size = tf.shape(X)[1]

        # Making embeddings for x
        embedder = layers.embedding_layer(X, vocab_dim=self.vocab_size, embedding_dim=self.embedding_dim, reuse=reuse)

        # Mask used to reset the query encoder when symbol is End-Of-Query symbol and to retain the state of the
        # session encoder when EoQ symbol has been seen yet.
        eoq_mask = tf.expand_dims(tf.cast(tf.not_equal(X, self.eoq_symbol), tf.float32), 2)

        # eoq mask has size [MAX LEN x BATCH SIZE] --> we want to loop over batch size
        # print(eoq_mask)

        # BATCH_SIZE = 80
        # MAX_LEN = 50  # TODO: this shouldn't be as local
        # for b in range(BATCH_SIZE):
        #     for qw in range(MAX_LEN):
        #         condition = tf.not_equal(eoq_mask[qw, b], 0, name='condition') # if not equal to zero --> then its not an eoq symbol
        #         # print(condition)
        #         # print(condition[0])
        #         a = tf.cond(condition[0], lambda: self.make_attention_mask(eoq=False), lambda: self.make_attention_mask(eoq=True))
        #         #print(a)



        # Computes the encoded query state. The tensorflow scan function repeatedly applies the gru_layer_with_reset
        # function to (embedder, eoq_mask) and it initialized the gru layer with the zero tensor.
        # In the query encoder we need the possibility to reset the gru layer, namely after the eos symbol has been
        # reached
        query_encoder_packed = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_reset(
                result_prev[1],  # h_reset_prev
                x,
                name='query_encoder',
                x_dim=self.embedding_dim,
                y_dim=self.query_dim,
                reuse=reuse
            ),
            (embedder, eoq_mask),  # scan does not accept multiple tensors so we need to pack and unpack
            initializer=tf.zeros((2, batch_size, self.query_dim))
        )

        query_encoder, hidden_query = tf.unpack(query_encoder_packed, axis=1)

        # This part does the same, yet for the session encoder. Here we need to have the possibility to keep the current
        # state where we were at, namely if we have not seen a full query. If we have, update the session encoder state.
        session_encoder_packed = tf.scan(
            lambda result_prev, x: layers.gru_layer_with_retain(
                result_prev[1],  # h_retain_prev
                x,
                name='session_encoder',
                x_dim=self.query_dim,
                y_dim=self.session_dim,
                reuse=reuse
            ),
            (query_encoder, eoq_mask),
            initializer=tf.zeros((2, batch_size, self.session_dim))
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
                x_dim=self.embedding_dim,
                h_dim=self.session_dim,
                y_dim=self.decoder_dim,
                reuse=reuse
            ),
            (embedder, eoq_mask, session_encoder),
            # scan does not accept multiple tensors so we need to pack and unpack
            initializer=tf.zeros((batch_size, self.decoder_dim))
        )

        # After the decoder we add an additional output layer
        flatten_decoder = tf.reshape(decoder, (-1, self.decoder_dim))
        flatten_embedder = tf.reshape(embedder, (-1, self.embedding_dim))
        # flatten_session_encoder = tf.reshape(session_encoder, (-1, self.session_dim))

        # attention

        # expand to batch_size x num_of_steps x query_dim
        # query_encoder_T = tf.transpose(query_encoder, perm=[1, 0, 2])
        # query_decoder_T = tf.transpose(decoder, perm=[1, 0, 2])

        # expand to num_of_steps x batch_size x num_of_steps x query_dim
        query_encoder_expanded = tf.tile(tf.expand_dims(query_encoder, 2), (1, 1, num_of_steps, 1))

        flatten_decoder_with_attention = \
            layers.attention(query_encoder_expanded, flatten_decoder, enc_dim=self.query_dim, dec_dim=self.decoder_dim,
                             reuse=reuse)

        output_layer = layers.output_layer(
            flatten_embedder,
            flatten_decoder_with_attention,
            x_dim=self.embedding_dim,
            h_dim=self.decoder_dim + self.query_dim,
            y_dim=self.output_dim,
            reuse=reuse
        )

        # We compute the output logits based on the output layer above
        flatten_logits = layers.logits_layer(
            output_layer,
            x_dim=self.output_dim,
            y_dim=self.vocab_size,
            reuse=reuse
        )

        logits = tf.reshape(flatten_logits, (num_of_steps, batch_size, self.vocab_size))

        # If we want the softmax back from this step or just the logits f or the loss function
        if return_softmax:
            output = self.softmax(logits)
        else:
            output = logits

        # If we want to continue decoding with single_step we need the hidden states of all GRU layers
        if return_last_with_hidden_states:
            hidden_decoder = decoder  # there is no resetted decoder output
            # Note for attention mechanism
            return output[-1, :, :], hidden_query[:, :, :], hidden_session[-1, :, :], hidden_decoder[-1, :, :]
        else:
            return output

    def single_step(self, X, prev_hidden_query_states, prev_hidden_session, prev_hidden_decoder, reuse=True):
        """
        Performs a step in the HRED X can be a 2-D tensor (batch, vocab), this can be used
        for beam search

        :param X: The input sessions. Lists of ints, ints correspond to words
        Shape: (max_length)
        :param start_hidden_query: The first hidden state of the query encoder. Initialized with zeros.
        Shape: (2 x query_dim)
        :param start_hidden_session: The first hidden state of the session encoder. Iniitalized with zeros.
        Shape: (2 x session_dim)
        :param start_hidden_decoder: The first hidden state of the decoder. Initialized with zeros.
        Shape: (output_dim)
        :return:
        """
        # Note that with the implementation of attention the object "prev_hidden_query_states" contains not only the
        # previous query encoded state but all previous states, therefore we need to get the last query state
        prev_hidden_query = prev_hidden_query_states[-1, :, :]
        # Making embeddings for x
        embedder = layers.embedding_layer(X, vocab_dim=self.vocab_size, embedding_dim=self.embedding_dim, reuse=reuse)

        # Mask used to reset the query encoder when symbol is End-Of-Query symbol and to retain the state of the
        # session encoder when EoQ symbol has been seen yet.
        eoq_mask = tf.cast(tf.not_equal(X, self.eoq_symbol), tf.float32)

        query_encoder, hidden_query = tf.unpack(layers.gru_layer_with_reset(
            prev_hidden_query,  # h_reset_prev
            (embedder, eoq_mask),
            name='query_encoder',
            x_dim=self.embedding_dim,
            y_dim=self.query_dim,
            reuse=reuse
        ))

        # This part does the same, yet for the session encoder. Here we need to have the possibility to keep the current
        # state where we were at, namely if we have not seen a full query. If we have, update the session encoder state.
        session_encoder, hidden_session = tf.unpack(layers.gru_layer_with_retain(
            prev_hidden_session,  # h_retain_prev
            (query_encoder, eoq_mask),
            name='session_encoder',
            x_dim=self.query_dim,
            y_dim=self.session_dim,
            reuse=reuse
        ))

        # This part makes the decoder for a step. The decoder uses both the word embeddings, the reset/retain vector
        # and the session encoder, so we give three variables to the decoder GRU. The decoder GRU is somewhat special,
        # as it incorporates the session_encoder into each hidden state update
        hidden_decoder = layers.gru_layer_with_state_reset(
            prev_hidden_decoder,
            (embedder, eoq_mask, session_encoder),
            name='decoder',
            x_dim=self.embedding_dim,
            h_dim=self.session_dim,
            y_dim=self.decoder_dim,
            reuse=reuse
        )

        decoder = hidden_decoder
        flatten_decoder = tf.reshape(decoder, (-1, self.decoder_dim))

        # add attention layer
        # expand to num_of_steps x batch_size x num_of_steps x query_dim
        num_of_atten_states = tf.shape(prev_hidden_query_states)[0]
        tf.Print(num_of_atten_states, [num_of_atten_states], "INFO - single-step ")
        tf.Print(flatten_decoder, [tf.shape(flatten_decoder)], "INFO - decoder.shape ")
        query_encoder_expanded = tf.tile(tf.expand_dims(prev_hidden_query_states, 2), (1, 1, num_of_atten_states, 1))

        flatten_decoder_with_attention = \
            layers.attention(query_encoder_expanded, flatten_decoder, enc_dim=self.query_dim, dec_dim=self.decoder_dim,
                             reuse=reuse)

        # After the decoder we add an additional output layer
        output = layers.output_layer(
            embedder,
            flatten_decoder_with_attention,
            x_dim=self.embedding_dim,
            h_dim=self.decoder_dim + self.query_dim,
            y_dim=self.output_dim,
            reuse=reuse
        )

        # We compute the output logits based on the output layer above
        logits = layers.logits_layer(
            output,
            x_dim=self.output_dim,
            y_dim=self.vocab_size,
            reuse=reuse
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
        labels = tf.one_hot(labels, self.vocab_size)

        eos_mask = tf.expand_dims(tf.cast(tf.not_equal(X, self.eos_symbol), tf.float32), 2)
        labels = labels * eos_mask

        loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        )

        tf.scalar_summary("loss", loss)
        return loss

    def non_padding_accuracy(self, logits, labels):
        """
        Accuracy on non padding symbols
        """

        # Matching non-padding tokens
        correct_prediction = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(np.argmax(self.softmax(logits)), labels),
                    tf.not_equal(tf.constant(self.eos_symbol, tf.int64), labels)
                ),
                tf.float32
            )
        )

        # All non-padding tokens
        not_padding_sum = tf.reduce_sum(
            tf.cast(
                tf.not_equal(tf.constant(self.eos_symbol, tf.int64), labels),
                tf.float32
            )
        )

        non_padding_acc = tf.div(correct_prediction, not_padding_sum)
        tf.scalar_summary("non-padding-acc", non_padding_acc)

        return non_padding_acc

    def non_symbol_accuracy(self, logits, labels):
        """
        Accuracy without padding, unknown and end-of-query symbol
        """

        # Matching non-symbol tokens (so all actual matching words)
        correct_prediction = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(np.argmax(self.softmax(logits)), labels),
                    tf.logical_and(
                        tf.not_equal(tf.constant(self.eos_symbol, tf.int64), labels),
                        tf.logical_and(
                            tf.not_equal(tf.constant(self.eoq_symbol, tf.int64), labels),
                            tf.not_equal(tf.constant(self.unk_symbol, tf.int64), labels)
                        )
                    )
                ),
                tf.float32
            )
        )

        # Sum all non-symbol tokens (so all actual words)
        not_symbol_sum = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.logical_and(
                        tf.not_equal(tf.constant(self.eos_symbol, tf.int64), labels),
                        tf.not_equal(tf.constant(self.eoq_symbol, tf.int64), labels)
                    ),
                    tf.not_equal(tf.constant(self.unk_symbol, tf.int64), labels)
                ),
                tf.float32
            )
        )

        non_symbol_acc = tf.div(correct_prediction, not_symbol_sum)
        tf.scalar_summary("non-symbol-acc", non_symbol_acc)

        return non_symbol_acc
