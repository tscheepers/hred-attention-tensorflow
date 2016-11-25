import numpy as np

from hred import HRED
import tensorflow as tf
import read_data

TRAIN_FILE = '../../data/tr_session.out'
VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
SMALL_FILE = '../../data/small_test_session.out'
eos = '2'

TRAIN_TFR = '../../data/tfrecords/train.tfrecords'
VALIDATION_TFR = '../../data/tfrecords/valid.tfrecords'
TEST_TFR = '../../data/tfrecords/test.tfrecords'
SMALL_TFR = '../../data/tfrecords/small.tfrecords'

if __name__ == '__main__':
    with tf.Graph().as_default():

        hred = HRED()
        batch_size = None
        max_length = None

        X = tf.placeholder(tf.int32, shape=(max_length, batch_size))
        HQ0 = tf.placeholder(tf.float32, (2, batch_size, hred.query_hidden_size))
        HS0 = tf.placeholder(tf.float32, (2, batch_size, hred.session_hidden_size))
        HD0 = tf.placeholder(tf.float32, (batch_size, hred.decoder_hidden_size))
        O0 = tf.placeholder(tf.float32, (batch_size, hred.vocab_size))

        graph = hred.step(X, HQ0, HS0, HD0, O0)

        label, input = read_data.read_and_decode(TEST_TFR)
        print input
        input_batch, labels_batch = tf.train.shuffle_batch(
            [input, label], batch_size=batch_size,
            capacity=2000,
            min_after_dequeue=1000)

        inp_b = tf.cast(input_batch, tf.int32)
        lab_b = tf.cast(labels_batch, tf.int32)
        print inp_b, lab_b

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter('logs/graph', sess.graph)

            # x = np.random.randint(0, hred.vocab_size, (5, 1))
            # print(x)

            # for serialized_example in tf.python_io.tf_record_iterator(TEST_TFR):
            #     example = tf.train.Example()
            #     example.ParseFromString(serialized_example)
            #
            #     input = example.features.feature['input'].int64_list.value
            #     label = example.features.feature['label'].int64_list.value
            #
            #     print input, label

            #
            # print input.va

            #print input_batch

            batch_size = 10
            max_length = 5

            summary_writer = tf.train.SummaryWriter('logs/graph', sess.graph)

            x = np.random.randint(0, hred.vocab_size, (max_length, batch_size))
            hq0 = np.zeros((2, batch_size, hred.query_hidden_size))
            hs0 = np.zeros((2, batch_size, hred.session_hidden_size))
            hd0 = np.zeros((batch_size, hred.decoder_hidden_size))
            o0 = np.zeros((batch_size, hred.vocab_size))

            # tf.train.start_queue_runners(sess=sess)
            # labels, images = sess.run([labels_batch, input_batch])


            softmax, output, decoder, session_encoder, query_encoder = sess.run(graph, {X:inp_b})#{X: input_batch})

            softmax = sess.run(
                graph,
                {X: x, HQ0: hq0, HS0: hs0, HD0: hd0, O0: o0}
            )

            print(softmax)
