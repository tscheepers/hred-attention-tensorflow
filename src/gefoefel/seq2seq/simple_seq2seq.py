import tensorflow as tf
from tensorflow.models.rnn.translate import seq2seq_model
import numpy as np
import random

# Parameters of the network --> need to tune this
SEQ_LENGTH = 5 #  note: for now i made fixed sizes
BATCH_SIZE = 64
MEMORY_DIM = 100
VOCAB_SIZE = 90004
EMBEDDING_DIM = 50
LEARNING_RATE = 0.05
MOMENTUM = 0.9
MAX_STEPS = 100

DATA_FILE_TRAIN = 'data/test_session.out' #  yes, this is test.. but for now :)
LOG_DIR = 'logs'


def train():

    # 1) Read in data
    x_train, y_train = read_data(DATA_FILE_TRAIN)

    # 2) Build the tensorflow graph
    with tf.Graph().as_default():
        enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" %t) for t in range(SEQ_LENGTH)]
        labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" %t) for t in range(SEQ_LENGTH)]
        weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]  # what weight initialization do we use??
        dec_inp =([tf.zeros_like(enc_inp[0], dtype=np.int32, name="BoS")] + enc_inp[:-1])
        prev_mem = tf.zeros((BATCH_SIZE, MEMORY_DIM))  # what's the effect of changing this?

        # We use a GRU cell, just like in the paper
        cell = tf.nn.rnn_cell.GRUCell(MEMORY_DIM)
        dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, VOCAB_SIZE, VOCAB_SIZE, EMBEDDING_DIM)  #  are these sizes correct?

        loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, VOCAB_SIZE)
        #acc = accuracy(dec_outputs, labels)
        tf.scalar_summary("loss", loss)

        summary = tf.merge_all_summaries()

        optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
        train_op = optimizer.minimize(loss)

        with tf.Session() as sess:

            summary_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph)
            sess.run(tf.initialize_all_variables())

            for i in range(MAX_STEPS):
                XY = zip(x_train, y_train)
                sample = random.sample(XY, BATCH_SIZE)
                X_train_batch, Y_train_batch = zip(*sample)

                feed_dict_train = {enc_inp[t]: X_train_batch[t] for t in range(SEQ_LENGTH)}
                feed_dict_train.update({labels[t]: Y_train_batch[t] for t in range(SEQ_LENGTH)})

                #print train_op, loss
                _, loss_train = sess.run([train_op, loss], feed_dict=feed_dict_train)
                #acc_train = sess.run(acc, feed_dict=feed_dict_train)
                print loss_train#, acc_train

                dec_outputs_batch = sess.run(dec_outputs, feed_dict_train)
                print [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]


def accuracy(dec_outputs, labels):
    print dec_outputs
    print labels
    correct_prediction = tf.equal(tf.argmax(dec_outputs, 1), tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.scalar_summary("accuracy", acc)
    return acc

def read_data(data_file):
    PAD = 3
    with open(data_file, 'r') as df:
        input_data = []
        labels = []
        for line in df:
            session = [map(int, x.split()) for x in line.strip().split('\t')]
            if len(session[0]) > PAD or len(session[1]) > PAD:
                continue
            else:
                padding_input = [3 for i in range(PAD - len(session[0]))]
                padding_output = [3 for i in range(PAD - len(session[1]))]
            input_data.append(session[0]+padding_input)
            labels.append(session[1]+padding_output)

    return np.array(input_data), np.array(labels)

if __name__ == '__main__':
    train()

#tf.nn.rnn_cell
# seq_length = 5
# batch_size = 64
# memory_dim = 100
# vocab_size = 7
# embedding_dim = 50

# with tf.Graph().as_default():
#     enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" %t) for t in range(seq_length)]
#     labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" %t) for t in range(seq_length)]
#     weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
#
#     dec_inp =([tf.zeros_like(enc_inp[0], dtype=np.int32, name="BoS")] + enc_inp[:-1])
#     prev_mem = tf.zeros((batch_size, memory_dim))
#
#     cell = tf.nn.rnn_cell.GRUCell(memory_dim)
#     dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim)
#
#     loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
#     tf.scalar_summary("loss", loss)
#
#     summary = tf.merge_all_summaries()
#
#     learning_rate = 0.05
#     momentum = 0.9
#     optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
#     train_op = optimizer.minimize(loss)
#
#     with tf.Session() as sess:
#         log_dir = 'logs'
#         summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)
#
#         sess.run(tf.initialize_all_variables())
#
#         for i in range(10):
#
#             X = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
#             Y = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]) #X[:]
#
#             feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
#             feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
#
#             _, loss_t = sess.run([train_op, loss], feed_dict)
#             print loss_t
#
#             dec_outputs_batch = sess.run(dec_outputs, feed_dict)
#             print [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]



