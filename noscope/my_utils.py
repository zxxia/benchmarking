
import numpy as np
import keras
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from keras.preprocessing import image
from collections import defaultdict, Counter

from keras.applications.resnet50 import preprocess_input



class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=128, dim=(224, 224), n_channels=3,
				 n_classes=10, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		# self.dataset = dataset
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size), dtype=int)
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			# img = imread(path + self.dataset + '/' + format(ID, '06d') + '.jpg')
			img =  image.load_img(ID, target_size= self.dim )
			# resized_img = resize(img, (self.dim[0], self.dim[1], self.n_channels))
			image.img_to_array(img)
			X[i,] = (image.img_to_array(img)).astype('float32') / 255
# resized_img
			# Store class
			img_name = ID.split('/')[-1]
			y[i] = self.labels[img_name]
		# X = preprocess_input(X)
		if self.n_classes == 1:
			return X, y
		else:
			return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
