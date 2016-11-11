from keras.models import Sequential
from keras.layers import Input, GRU, RepeatVector, TimeDistributed
from keras.layers import Dense, Dropout, Activation, Embedding, GRU
import numpy as np



# inputs = Input(shape=(timesteps, input_dim))
# encoded = LSTM(latent_dim)(inputs)

# decoded = RepeatVector(timesteps)(encoded)
# decoded = LSTM(input_dim, return_sequences=True)(decoded)

# sequence_autoencoder = Model(inputs, decoded)
# encoder = Model(inputs, encoded)



class EncoderDecoder:

	def __init__(self):
		self.model = Sequential()

	def add_embedding_layer(self, vocab_size, input_length, output_dim): 
		""" Vocab shape: number of words in your vocab+1
			Output dim: dimensions of the next layer
			Input length: length of input vector
		"""
		self.model.add(Embedding(vocab_size, input_length=input_length, output_dim=output_dim))

	def add_gru_layer(self, output_dim, return_sequences=False):
		""" Output dim: dimensions of the output
			Return sequences: false for encoder, true for decoder
		"""
		self.model.add(GRU(output_dim, return_sequences=return_sequences))

	def add_repeat_vector(self, rep):
		self.model.add(RepeatVector(rep))

	def add_dense_layer(self, output_dim, activation='softmax'):
		self.model.add(Dense(output_dim, activation=activation))

	def add_time_distribution(self, vocab_size, input_shape, activation='softmax'):
		self.model.add(TimeDistributed(Dense(vocab_size, input_shape=input_shape, activation=activation)))

	def compile_model(self):
		self.model.compile(optimizer='sgd',
          loss='mean_squared_error',
          metrics=['accuracy'])

	def train_model(self, x, y, nb_epoch):
		""" x = input data: numpy array or list of numpy arrays
				(if the model has multiple inputs) 
			y = labels: as a numpy array
		"""
		self.model.fit(x, y, nb_epoch=nb_epoch)

	def predict(self, x):
		self.output = self.model.predict(x)

	def evaluate_model(self, x, y):
		self.score = self.model.evaluate(x, y)

def encode_query(query, max_len, vocab_size):
	x = np.zeros((max_len, vocab_size))
	for i, q in enumerate(query):
		x[i][q] = 1
	return x


if __name__ == '__main__':

	vocab_size = 4+1#2+1
	max_len = 10 # as we have queries of variable length, we need padding
	output_dim_embedding = 2#64
	input_shape = (1,3)
	output_dim_gru = 64
	output_dim_dense = 5#3
	repeat = 3
	nb_epoch = 200
	
	train_queries = np.array([[1, 1, 1, 3, 3, 3], [1, 1]])
	target_queries = np.array([[2, 2, 2, 4, 4, 4], [2, 2]])
	
	# The time distributed dense needs a 3d array
	# Because we have queries of variable length, we need padding = max_len
	X_train = np.zeros((len(train_queries), max_len, vocab_size))
	for i, query in enumerate(train_queries):
		X_train[i] = encode_query(query, max_len, vocab_size)

	Y_train = np.zeros((len(target_queries), max_len, vocab_size))
	for i, query in enumerate(target_queries):
		Y_train[i] = encode_query(query, max_len, vocab_size)

	# Make an encoder-decoder
	ED = EncoderDecoder()
	ED.add_embedding_layer(vocab_size, max_len, output_dim_embedding)

	# Encode the input using a GRU
	ED.add_gru_layer(output_dim_gru, return_sequences=False)

	# This is gonna be the input of the decoder, repeat the input for each time step
	ED.add_repeat_vector(repeat)

	# Decode:
	ED.add_time_distribution(vocab_size, input_shape)
	
	ED.compile_model()

	ED.train_model(X_train, Y_train, nb_epoch)
	# ED.evaluate_model(X_train, Y_train)
	# print ED.score
	# print X_train
	# print Y_train
	# ED.predict(X_train)
	# print ED.output



	# X_train = np.array([np.eye(vocab_size)[x] for x in train_queries])
	# Y_train = np.array([np.eye(vocab_size)[x] for x in target_queries])
	# print X_train
	# Make an encoder
	
	# E = EncoderDecoder()
	# E.add_embedding_layer(vocab_size, output_dim_embedding)

	# E.add_gru_layer(output_dim_gru, return_sequences=False)
	# E.add_dense_layer(output_dim_dense)
	# E.compile_model()
	# E.train_model(X_train, X_train, nb_epoch)
	# E.evaluate_model(X_train, Y_train)
	# print E.score
	# print X_train
	# print Y_train
	# E.predict(X_train)
	# print E.output

	# # Make a decoder
	# D = EncoderDecoder()
	# D.add_embedding_layer(vocab_size, output_dim_embedding)

	# D.add_gru_layer(output_dim_gru, input_shape, return_sequences=False)
	
	# D.add_dense_layer(output_dim_dense)
	# D.compile_model()
	# D.train_model(X_train, Y_train, nb_epoch)
	# D.evaluate_model(X_train, Y_train)
	# print D.score
	# print X_train
	# print Y_train
	# D.predict(X_train)
	# print D.output

	