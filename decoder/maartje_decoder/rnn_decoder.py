from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, GRU
import numpy as np

class Model:

	def __init__(self):
		self.model = Sequential()

	def add_embedding_layer(self, vocab_shape, output_dim): #, input_length):
		""" Vocab shape: number of words in your vocab+1
			Output dim: dimensions of the next layer
			Input length: length of input vector
		"""
		self.model.add(Embedding(vocab_shape, output_dim=output_dim))#, input_length=input_length))
		print self.model.get_config()

	def add_gru_layer(self, output_dim, input_length=None): #, init):
		""" Output dim: dimensions of the output
		"""
		self.model.add(GRU(output_dim))#, input_length=input_length))

	def add_dense_layer(self, output_dim, activation='softmax'):
		self.model.add(Dense(output_dim, activation=activation))

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






if __name__ == '__main__':

	vocab_shape = 2+1#5+1 
	output_dim_embedding = 64
	output_dim_gru = 64
	output_dim_dense = 3#1
	nb_epoch = 200
	
	train_queries = np.array([1, 1, 1])
	target_queries = np.array([2, 2, 2])
	X_train = np.array([np.eye(vocab_shape)[x] for x in train_queries])
	Y_train = np.array([np.eye(vocab_shape)[x] for x in target_queries])

	M = Model()
	M.add_embedding_layer(vocab_shape, output_dim_embedding)
	M.add_gru_layer(output_dim_gru)
	M.add_dense_layer(output_dim_dense)
	M.compile_model()
	M.train_model(X_train, Y_train, nb_epoch)
	M.evaluate_model(X_train, Y_train)
	print M.score
	print X_train
	print Y_train
	M.predict(X_train)
	print M.output
