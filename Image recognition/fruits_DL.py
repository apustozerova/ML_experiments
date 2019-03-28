import os
import argparse
import csv
import datetime
import glob
import math
import sys
import time
import numpy as np
import imghdr

from resizeimage import resizeimage
from PIL import Image
from theano import config
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit

import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization

from PIL import Image
from theano import config
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from utilityF import plot_confusion_matrix, log_model_summary, plot_history
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


from keras.preprocessing.image import ImageDataGenerator

import fruit_cnn_config


np.random.seed(42) # initialize a random seed here to make the experiments repeatable with same results

# -------------------------READING PARAMETERS----------------------------------------------


def main(imagePath):
	images_path = []
	target_labels = []

	for root, dirs, files in os.walk(imagePath):
		for file in files:
			if imghdr.what(os.path.join(root, file))=='jpeg': #only "jpeg" files
				classes = root.replace("FIDS30/", "")
				target_labels.append(classes)
				images_path.append(os.path.join(root, file))

	#-------------------------------------------------------------------------------------
	#--------------------- Find all the classes and put targets for into array -----------

	le = preprocessing.LabelEncoder()
	le.fit(target_labels) # this basically finds all unique class names, and assigns them to the numbers
	# now we transform our labels to integers
	target = le.transform(target_labels); 
	target = np.asarray(target)#target put into array
	nclasses = len(le.classes_)
	print("Found the following "+ str(nclasses) + " classes: " + str(list(le.classes_)))

	#-------------------------------------------------------------------------------------
	#--------------------- Load the images and put into one array ------------------------
	images = []

	for filename in images_path:
		with Image.open(filename) as img:
			img = resizeimage.resize_contain(img, [250, 150]) #reshape all the images to one frame
			images.append(np.array(img)) # we convert the images to a Numpy array and store them in a list

	# a list of many 250x150 images is made into 1 big array
	img_array = np.array(images, dtype=config.floatX)

	print('Shape of array with all the images',img_array.shape) # must be (951, 150, 250, 3) 


	# -------------- Shuffling all the images and splittin to training and test set--------

	ss = ShuffleSplit(n_splits=1, test_size=0.2)
	random_splits = [(train_index, test_index) for train_index, test_index in ss.split(img_array)]

	train,test = random_splits[0]

	img_array_train = img_array[train]
	img_array_test = img_array[test]

	classes = target[train]
	test_classes = target[test]

	# ------------------------ Normalization ------------------------------------------

	print('Training set normalization')
	mean_train = img_array_train.mean()
	stddev_train = img_array_train.std()
	print('Mean and stdev before normalization',mean_train, stddev_train)

	img_array_norm_train = (img_array_train - mean_train) / stddev_train
	print('After normalization',img_array_norm_train.mean(), img_array_norm_train.std())

	print('Test set normalization')
	mean_test = img_array_test.mean()
	stddev_test = img_array_test.std()
	print('Mean and stdev before normalization',mean_test, stddev_test)

	img_array_norm_test = (img_array_test - mean_test) / stddev_test
	print('After normalization',img_array_norm_test.mean(), img_array_norm_test.std())


	# ------------------------ Flatten training images to vectors-----------------------------

	images_flat_train = img_array_norm_train.reshape(img_array_train.shape[0],-1)
	print("Shape of training set after flattering",images_flat_train.shape)

	input_shape_train = images_flat_train.shape[1]

	images_flat_test = img_array_norm_test.reshape(img_array_test.shape[0],-1)
	print("Shape of test set after flattering",images_flat_test.shape)

	input_shape_test = images_flat_test.shape[1]

	#----------------------------------------------------------------------------------------
	#-------------------------- Convolutional Neural Network---------------------------------



	print("Convolutional Neural Network...")
	print("[Start] Convolutional Neural Network\n")

	n_channels = 3 

	if keras.backend.image_dim_ordering() == 'th':
	    # Theano ordering (~/.keras/keras.json: "image_dim_ordering": "th")
	    train_img = img_array_train.reshape(img_array_train.shape[0], n_channels, img_array_train.shape[1], img_array_train.shape[2])
	    test_img = test_images.reshape(img_array_test.shape[0], n_channels, img_array_test.shape[1], img_array_test.shape[2])
	else:
	    # Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
	    train_img = img_array_train.reshape(img_array_train.shape[0], img_array_train.shape[1], img_array_train.shape[2], n_channels)
	    test_img = img_array_test.reshape(img_array_test.shape[0], img_array_test.shape[1], img_array_test.shape[2], n_channels)

	input_shape = train_img.shape[1:] 
	#print("\n\n" + str(input_shape)+ "\n\n")


	# Print CNN architures to be tested
	print("Models used for testing:\n")
	i = 1
	for model in fruit_cnn_config.cnn_to_test:
		print("\n[CNN"+ str(i) +"]\n")
	# 	log_model_summary(logs_file, model)
		i = i + 1


	# Compiling the model
	loss = fruit_cnn_config.loss
	epochs = fruit_cnn_config.epochs
	print("Loss: " +  loss + " -- Epochs: " + str(epochs) +"\n")

	#
	tr_times = []
	test_times = []
	# Save params for best accuracy as : acc value, model number, model and optimizer
	best_acc = (0,0,fruit_cnn_config.cnn_to_test[0], '')
	for optimizer in fruit_cnn_config.optimizers:
		print("\nOptimizer: " + optimizer +  "\n")
		i = 0
		for model in fruit_cnn_config.cnn_to_test:

			# Training showing values for portion of validation data
			start = time.time()
			model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
			history = model.fit(train_img, classes, batch_size=32, epochs=epochs, validation_split=0.1) 
			end = time.time()
			tr_times.append(end - start)

			# Testing for test images
			start = time.time()
			predictions = model.predict_classes(test_img)
			acc = accuracy_score(test_classes, predictions)
			end = time.time()
			test_times = end - start

			print("Accuracy on test set: " + str(acc) + " with cnn"+ str(i) +  " and optimizer " + optimizer + "\n")
			print("Accuracy on test set: " + str(acc) + " with cnn"+ str(i) +  " and optimizer " + optimizer + "\n")
			if fruit_cnn_config.isToSaveImages == True:
				plot_history(history.history, epochs, '../history_fruit_cnn_'+str(i)+ '_' + optimizer + '.jpg')

			if acc > best_acc[0]:
				best_acc = (acc, i, model, optimizer)

			i=i+1


	print("Best overall accuracy(" + str(best_acc[0])  +") for model cnn"  + str(best_acc[1]) + " and optmizer " + best_acc[3] +"\n")
	print("\nTraining times (mean): "  + str(np.mean(tr_times)) + "\n")
	print("Testing times (mean): "  + str(np.mean(test_times)) + "\n")

	# CM for best results
	model = best_acc[2]
	predictions = model.predict_classes(test_img)
	cm = confusion_matrix(test_classes, predictions, [0, 1])
	if fruit_cnn_config.isToSaveImages == True:
		plot_confusion_matrix(cm, classes={0,1}, title='Confusion matrix for fruit dataset with CNN' + str(best_acc[1]), filename='../../cm_fruit_cnn.jpg' )

	print("[End] Convolutional Neural Network\n")


	#--------------------------------------------------------------------
	# ------------- Data Augmentation -----------------------------------


	print("CNN  with Data Augmentation...")
	print("[Start] CNN  with Data Augmentation\n")

	datagen = ImageDataGenerator(
	    rotation_range=20,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=True)


	np.random.seed(42) # enforce repeatable result

	tr_times = []
	test_times = []


	# Split 
	rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=42)
	train_i, val_i = next(rs.split(train_img))

	# Images for training and testing
	s_train_img = train_img[train_i]
	s_val_img = train_img[val_i]

	# Label for training and testing
	# ImageDataGenerator needs the classes as Numpy array instead of normal list
	s_classes_train_array = np.array(classes)[train_i]
	s_classes_val_array = np.array(classes)[val_i]

	validation_data = (s_val_img, s_classes_val_array)


	# Save params for best accuracy as : acc value, model number, model and optimizer
	best_acc = (0,0,fruit_cnn_config.cnn_to_test[0], '')
	for optimizer in fruit_cnn_config.optimizers:
		print("Optimizer: " + optimizer +  "\n")
		i = 0
		for model in fruit_cnn_config.cnn_to_test:
			start = time.time()
			model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

			# fits the model on batches with real-time data augmentation:
			history = model.fit_generator(datagen.flow(s_train_img, s_classes_train_array, batch_size=16),
					    samples_per_epoch=len(s_train_img), epochs=epochs,
					    validation_data=validation_data)
			end = time.time()
			tr_times.append(end - start)

			start = time.time()
			test_pred = model.predict_classes(test_img)
			predictions  = model.predict_classes(test_img)
			acc = accuracy_score(test_classes, predictions)
			end = time.time()
			test_times.append(end - start)

			print("Accuracy on test set: " + str(acc) + " with cnn"+ str(i) +  " and optimizer " + optimizer + "\n")
			print("Accuracy on test set  " + str(acc) + " with cnn"+ str(i) +  " and optimizer " + optimizer +"\n")

			if fruit_cnn_config.isToSaveImages == True:
				plot_history(history.history, epochs, '../history_fruit_cnn_'+ str(i)+ '_' + optimizer +'_augmented.jpg')

			if acc > best_acc[0]:
				best_acc = (acc, i, model, optimizer)

			i=i+1


	print("Best overall accuracy(" + str(best_acc[0])  +") for model cnn"  + str(best_acc[1]) + " and optmizer " + best_acc[3] +"\n")
	print("\nTraining times (mean): "  + str(np.mean(tr_times)) + "\n")
	print("Testing times (mean): "  + str(np.mean(test_times)) + "\n")

	# CM for best results
	model = best_acc[2]
	predictions = model.predict_classes(test_img)
	cm = confusion_matrix(test_classes, predictions, [0, 1])
	if fruit_cnn_config.isToSaveImages == True:
		plot_confusion_matrix(cm, classes={0,1}, title='Confusion matrix for fruit dataset with CNN' + str(best_acc[1]) + ' and data augmentation', filename='../../cm_fruit_cnn_augment.jpg' )


	print("[Finish] CNN  with Data Augmentation\n")




	# ---------------------------------------------------------------------
	# ------ Simple fully connected network.. -----------------------------

	print("Simple fully connected network...")
	print("[Start] Simple fully connected network\n")

	input_shape = images_flat_train.shape[1]
	# simple Fully-connected network
	model = Sequential()
	model.add(Dense(128, input_dim=input_shape))
	model.add(Dense(256))
	model.add(Dense(30,activation='softmax'))

	# log_model_summary(logs_file, model)

	# Optimizer = Stochastic Gradient Descent
	optimizer = 'sgd' 

	print("Loss: " +  loss + " -- Optimizer: " + optimizer + "\n")

	# Compiling the model
	# This creates the whole model structure in memory. 
	start = time.time()
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

	# Training
	model.fit(images_flat_train, classes, batch_size=32, epochs=epochs) #, validation_data=validation_data)
	end = time.time()
	print("Training time: " + str(end - start) + "\n")

	# verify Accuracy on Train set
	start = time.time()
	predictions = model.predict_classes(images_flat_train)
	acc = accuracy_score(classes, predictions)
	end = time.time()
	print("Accuracy on training set: " + str(acc) + "\n")
	print("Testing time: " + str(end - start) + "\n")


	# Predictions
	start = time.time()
	test_pred = model.predict_classes(images_flat_test)
	acc = accuracy_score(test_classes, test_pred)
	end = time.time()
	print("Accuracy on test set: " + str(acc) + "\n")
	print("Training time: " + str(end-start) + "\n")
	print("[Finish] Simple fully connected network\n")



	# logs_file.close()


