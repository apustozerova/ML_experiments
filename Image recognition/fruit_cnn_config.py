import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization


# Configuration file for running experiments of Neural Networks with fruits dataset

# List of optimizers test
optimizers = ['sgd', 'adam']
# Number of epochs 
epochs = 15
# Define a loss function 
loss = 'sparse_categorical_crossentropy' 


input_shape = (150, 250, 3)

# From tutorial
cnn0 = Sequential([
	# Layer 1
	Conv2D(16, 3, padding = 'valid', activation='relu', input_shape=input_shape),
	BatchNormalization(),
	# reduces image size by half
	MaxPooling2D(pool_size=(2, 2)),
	# random "deletion" of %-portion of units in each batch
	# Dropout randomly switches off neurons, which prevents overfitting
	Dropout(0.3), 

	# Layer 2
	Conv2D(16, 3),
	#model.add(BatchNormalization())
	Activation('relu'),
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	Dropout(0.3),
	Flatten(),

	# Full Layer
	Dense(256),
	Activation('relu'),
	Dropout(0.1),

	#For class prediction
	Dense(30,activation='softmax')
])



# With 32 channels on the second layer
cnn1 = Sequential([
	# Layer 1
	Conv2D(16, 3, padding = 'valid', activation='relu', input_shape=input_shape),
	BatchNormalization(),
	# reduces image size by half
	MaxPooling2D(pool_size=(2, 2)),
	
	# Layer 2
	Conv2D(32, 3),
	#model.add(BatchNormalization())
	Activation('relu'),
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	Flatten(),

	# Full Layer
	Dense(256),
	Activation('relu'),

	#For class prediction
	Dense(30,activation='softmax')
])


# Three layers with 16-32-32 channels
cnn2 = Sequential([
	# Layer 1
	Conv2D(16, 3, padding = 'valid', activation='relu', input_shape=input_shape),
	BatchNormalization(),
	# reduces image size by half
	MaxPooling2D(pool_size=(2, 2)),
	# random "deletion" of %-portion of units in each batch
	# Dropout randomly switches off neurons, which prevents overfitting
	Dropout(0.3), 

	# Layer 2
	Conv2D(32, 3, padding = 'valid'),
	#model.add(BatchNormalization())
	MaxPooling2D(pool_size=(2, 2)),
	Dropout(0.3),
    
	# Layer 3
	Conv2D(32, 3),
	#model.add(BatchNormalization())
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	Dropout(0.3),
	Flatten(),

	# Full Layer
	Dense(256),
	Dropout(0.1),

	#For class prediction
	Dense(30,activation='softmax')
])

cnn_to_test = [cnn0, cnn1, cnn2]


# ------ Extra debug information
# Set to True if you want to save images of hystory plots and confusion matrix
isToSaveImages = False

