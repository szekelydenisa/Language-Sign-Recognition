# import the necessary packages
from keras.models import Sequential
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K

class SmallVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape
		model = Sequential()
		inputShape = (height, width, depth)

		# CONV => RELU => POOL layer set
		model.add(Conv2D(16, (2,2), input_shape=inputShape, activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
		model.add(Conv2D(32, (3,3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
		model.add(Conv2D(64, (5,5), activation='relu'))
		model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(classes, activation='softmax'))
	
		sgd = optimizers.SGD(lr=1e-2)
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		filepath="cnn_model_keras.h5"
		checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint1]
		

    	# return the constructed network architecture
		return model, callbacks_list





