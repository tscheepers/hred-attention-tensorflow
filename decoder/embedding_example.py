import itertools
import numpy as np

sentences = '''
sam is red
hannah not red
hannah is green
bob is green
bob not red
sam not green
sarah is red
sarah not green'''.strip().split('\n')
is_green = np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype='int32').T

lemma = lambda x: x.strip().lower().split(' ')
sentences_lemmatized = [lemma(sentence) for sentence in sentences]
words = set(itertools.chain(*sentences_lemmatized))
# {'hannah', 'not', 'sam', 'is', 'red', 'green', 'bob', 'sarah'}

# dictionaries for converting words to integers and vice versa
word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

# convert the sentences a numpy array
to_idx = lambda x: [word2idx[word] for word in x]
sentences_idx = [to_idx(sentence) for sentence in sentences_lemmatized]
sentences_array = np.asarray(sentences_idx, dtype='int32')

# parameters for the model
sentence_maxlen = 3
n_words = len(words)
n_embed_dims = 3

# put together a model to predict
from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN
from keras.models import Model
import keras.backend as K

input_sentence = Input(shape=(sentence_maxlen,), dtype='int32')
input_embedding = Embedding(n_words, n_embed_dims)(input_sentence)
color_prediction = SimpleRNN(1)(input_embedding)

predict_green = Model(input=[input_sentence], output=[color_prediction])
predict_green.compile(optimizer='sgd', loss='binary_crossentropy')

# fit the model to predict what color each person is
predict_green.fit([sentences_array], [is_green], nb_epoch=1000, verbose=1)
embeddings = K.get_value(predict_green.layers[1].W)

# print out the embedding vector associated with each word
for i in range(n_words):
	print('{}: {}'.format(idx2word[i], embeddings[i]))