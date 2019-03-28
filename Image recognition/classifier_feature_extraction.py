


# --- Utility functions
from utilityF import plot_confusion_matrix
import classifier_feature_extraction_config




import argparse
import os
import glob, sys
import time
import numpy as np
import datetime
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier




def main(imagePath, algo_list, dataset, n_bins):
	# create file for logs
	logs_file = open('log_'+ dataset +'.text','w') 
	logs_file.write("Arguments: " + imagePath + " -- " + str(algo_list) + " -- " + str(n_bins) + " -- " + dataset +"\n")

	# ----------------------------EXTRACTING GROUND TRUTH--------------------------------------


	# Find all images in that folder
	os.chdir(imagePath)
	fileNames = glob.glob("TrainImages/*.pgm") if dataset == 'car' else glob.glob("*/*.jpg")
	numberOfFiles=len(fileNames)
	targetLabels=[]

	print("Found " + str(numberOfFiles) + " files\n")
	logs_file.write("Found " + str(numberOfFiles) + " files\n")

	# The first step - create the ground truth (label assignment, target, ...) 
	# For that, iterate over the files, and obtain the class label for each file
	# Basically, the class name is in the full path name, so we simply use that

	print("Start label encoding...")
	logs_file.write("[Start] Creating label encoding\n")
	start = time.time()

	if dataset == 'fruit':
		for fileName in fileNames:
			if sys.platform.startswith('win') or sys.platform.startswith('cygwin'):
				pathSepIndex = fileName.index("\\")
				targetLabels.append(fileName[:pathSepIndex])
			else:
				pathSepIndex = fileName.index("/")						
				targetLabels.append(fileName[:pathSepIndex])
		# sk-learn can only handle labels in numeric format - we have them as strings though...
		# Thus we use the LabelEncoder, which does a mapping to Integer numbers
		from sklearn import preprocessing
		le = preprocessing.LabelEncoder()
		le.fit(targetLabels) # this basically finds all unique class names, and assigns them to the numbers
		logs_file.write("Found the following classes: " + str(list(le.classes_)) +"\n")

		# now we transform our labels to integers
		target = le.transform(targetLabels); 
		# print("Transformed labels (first elements: " + str(target[0:150]))

		# If we want to find again the label for an integer value, we can do something like this:
		# print list(le.inverse_transform([0, 18, 1]))
	else:
		for fileName in fileNames:
			if fileName.startswith('TrainImages/neg'):
				targetLabels.append(0)
			elif fileName.startswith('TrainImages/pos'):
				targetLabels.append(1)
			else:
				print("Cannot extract ground truth. Format error in file names.") 
			target = targetLabels


	end = time.time()
	logs_file.write("[Finish] Creating label encoding. Elapsed time: " + str(end - start) + "\n")


	# --------------------FEATURE EXTRACTION------------------------------------------------
	

	print("Extracting Features...\n")
	logs_file.write("[Start] Extracting Features\n")
	start = time.time()

	# use our own simple function to flatten the 2D arrays
	flatten = lambda l: [item for sublist in l for item in sublist]

	data=[]
	descriptors = []
	chists = []
	for index, fileName in enumerate(fileNames):

		#print(imagePath + "/" + (fileName.replace('\\', '/')))

		# colour histogram

		imageOpenCV = cv2.imread(imagePath + "/" + (fileName.replace('\\', '/'))) 
	
		if imageOpenCV is not None:
	
	
			# print(imageOpenCV.shape[0])
			# print(imageOpenCV.shape[1])
		
			features = []	
			gray= cv2.cvtColor(imageOpenCV,cv2.COLOR_BGR2GRAY)	
			chans = cv2.split(imageOpenCV) # split the image in the different channels (RGB, but in open CV, it is BGR, actually..)
			if dataset == 'fruit':
				# resize, if height or width is above 3000 (important for SIFT) to height = 1000 
				height = imageOpenCV.shape[0]
				width = imageOpenCV.shape[1]
				if height > 2000 or width > 2000:
					ratio = float(height)/float(width)
					imageOpenCV = cv2.resize(imageOpenCV,(1000, int(ratio * 1000)), interpolation = cv2.INTER_AREA)

				colors = ("b", "g", "r")
				# loop over the image channels
				for (chan, color) in zip(chans, colors):
					# create a histogram for the current channel and concatenate the resulting histograms for each channel
					hist = cv2.calcHist([chan], [0], None, [n_bins], [0, 256])
					features.extend(hist)
			else:
				# loop over the image channels
				for chan in chans:
					# create a histogram for the current channel and concatenate the resulting histograms for each channel
					hist = cv2.calcHist([chan], [0], None, [n_bins], [0, 256])
					features.extend(hist)
			
			features = flatten(features) # and append this to our feature vector
			#chists.append(features)
		
			# SIFT 
			sift = cv2.xfeatures2d.SIFT_create()
			kp, des = sift.detectAndCompute(gray,None)		# get SIFT key points and descriptors
		
			if des is not None: 
				descriptors.append(des)
				chists.append(features)
			else:
				print(imagePath + "/" + (fileName.replace('\\', '/')) + " failed!")
				target = np.delete(target, index)
	
		else:
			msg = imagePath + "/" + (fileName.replace('\\', '/')) + " failed!\n";
			#print(msg)
			logs_file.write(msg)
			target = np.delete(target, index)

	end = time.time()
	logs_file.write("[Finish] Extracting Features. Elapsed time: " + str(end - start) + "\n\n")

	#------------------------------ BAG OF VISUAL WORDS  - HELPERS -----------------------------------------------
	# from kushalvyas.github.io, adapted

	

	def formatND(l):
		"""	
		restructures list into vstack array of shape
		M samples x N features for sklearn
		"""
		vStack = np.array(l[0])
		for remaining in l[1:]:
			vStack = np.vstack((vStack, remaining))
		return vStack
	
	def cluster(vStack, n_clusters):
		"""	
		cluster using KMeans algorithm, 
		"""
		kmeans_obj = KMeans(n_clusters = n_clusters)
		kmeans_ret = kmeans_obj.fit_predict(vStack)	
		return kmeans_ret

	def developVocabulary(n_images, descriptor_list, n_clusters, kmeans_ret):
		"""
		Each cluster denotes a particular visual word 
		Every image can be represeted as a combination of multiple 
		visual words. The best method is to generate a sparse histogram
		that contains the frequency of occurence of each visual word 
		Thus the vocabulary comprises of a set of histograms of encompassing
		all descriptions for all images
		"""
	
		mega_histogram = np.array([np.zeros(n_clusters) for i in range(n_images)])
		old_count = 0
		for i in range(n_images):
			l = len(descriptor_list[i])
			for j in range(l):
				idx = kmeans_ret[old_count+j]
				mega_histogram[i][idx] += 1
			old_count += l
		print("Vocabulary Histogram Generated")
		return mega_histogram
	
	def standardize(hist, std=None):
		"""
	
		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.
		"""
		if std is None:
			scale = StandardScaler().fit(hist)
			hist = scale.transform(hist)
		else:
			print("STD not none. External STD supplied")
			hist = std.transform(hist)
		
		return hist	
		
	#------------------------------ BAG OF VISUAL WORDS -----------------------------------------------
	# from kushalvyas.github.io, adapted


	# training the BoVW
	print("Computing Bag of visual Words...")
	logs_file.write("[Start] Computing Bag of Visual Words\n")
	start = time.time()
	n_clusters = 20
	print("Changing format...")
	des_stack = formatND(descriptors)
	print("Clustering...")
	km_ret =  cluster(des_stack, n_clusters)
	print("Developing the vocabulary...")											
	histogram = developVocabulary(n_images = len(descriptors), descriptor_list=descriptors, n_clusters = n_clusters, kmeans_ret = km_ret)
	print("Standardizing...")
	histogram = standardize(histogram)

	# append the BoVW with the colour histogram
	for i in range(len(chists)):
		row = chists[i] 
		row.extend(histogram[i].tolist())
		data.append(row)

	end = time.time()	
	logs_file.write("[Finish] Computing Bag of Visual Words. Elapsed time: " + str(end - start) + "\n\n")


	#-------------------------------- CLASSIFICATION -------------------------------------------------
	
	# Data to numpy array and creation of split for cross validation
	data = np.array(data)

	# Separates data for training and for validation
	data, x_val, target, y_val = train_test_split(data, target, test_size=0.2, random_state=42, stratify=target)

	kf = StratifiedKFold(n_splits=3, random_state =42)
	splits = []
	for train_index, test_index in kf.split(data,target):
		splits.append((train_index, test_index))

	logs_file.write("[Data shape] " + str(data.shape) + "\n")

	# MLP -------------------
	if "mlp" in algo_list:

		logs_file.write("[Start] MLP Parameter tunning \n")
		training_time = []
		testing_time = []
		print("[MLP] Parameter tunning")
		# parameter selection with grid search for the number of nodes
		# taking only one hidden layer
		best_size = (0, 0)
		for size in classifier_feature_extraction_config.mlp_size:
			acc = []
			for (train_index, test_index) in splits:
				#logs_file.write(str(train_index) + " --- " + str(test_index) + "\n")
				x_train, x_test = data[train_index], data[test_index]
				y_train, y_test = target[train_index], target[test_index]
				start_tr = time.time()
				model = MLPClassifier(hidden_layer_sizes= (size,))
				model.fit(x_train, y_train)
				end_tr = time.time()
				training_time.append(end_tr - start_tr)

				mean_acc = model.score(x_test, y_test)
				acc.append(mean_acc)
				end_test = time.time()
				testing_time.append(end_test - end_tr)
		
			mean_acc = np.mean(acc)
			logs_file.write("Mean accuracy for number of nodes="+str(size) + ":  " + str(mean_acc) + " with std " + str(np.std(acc)) +"\n")
			if mean_acc > best_size[1]:
				best_size = (size, mean_acc)
	
		# parameter selection for activation function
		best_act = ('', 0)
		for act in classifier_feature_extraction_config.mlp_activation:
			acc = []
			for (train_index, test_index) in splits:
				x_train, x_test = data[train_index], data[test_index]
				y_train, y_test = target[train_index], target[test_index]

				start_tr = time.time()
				model = MLPClassifier(activation = act)
				model.fit(x_train, y_train)
				end_tr = time.time()
				training_time.append(end_tr - start_tr)

				mean_acc = model.score(x_test, y_test)
				acc.append(mean_acc)
				end_test = time.time()
				testing_time.append(end_test - end_tr)

			mean_acc = np.mean(acc)
			logs_file.write("Mean accuracy for activation="+ act + ":  " + str(mean_acc) + " with std " + str(np.std(acc)) +"\n")
			if mean_acc > best_act[1]:
				best_act = (act, mean_acc)
			
			
		# parameter selection for solver
		best_solver = ('', 0)
		for solv in classifier_feature_extraction_config.mlp_solver:
			acc = []
			for (train_index, test_index) in splits:
				x_train, x_test = data[train_index], data[test_index]
				y_train, y_test = target[train_index], target[test_index]
				start_tr = time.time()
				model = MLPClassifier(solver = solv)
				model.fit(x_train, y_train)
				end_tr = time.time()
				training_time.append(end_tr - start_tr)

				mean_acc = model.score(x_test, y_test)
				acc.append(mean_acc)
				end_test = time.time()
				testing_time.append(end_test - end_tr)

			mean_acc = np.mean(acc)
			logs_file.write("Mean accuracy for solver="+ solv + ":   " + str(mean_acc) + " with std " + str(np.std(acc)) +"\n")
			if mean_acc > best_solver[1]:
				best_solver = (solv, mean_acc)

		logs_file.write("[Finish] MLP Parameter tunning \n")
		logs_file.write("Training times mean: " + str(np.mean(training_time)) + "\n" ) 
		logs_file.write("Testing times mean: " + str(np.mean(testing_time)) + "\n" ) 

		logs_file.write("[Start] MLP Validation\n")
		# validation
		msg = "Taking: \n Best number of nodes:  "+ str(best_size[0]) + ", best activation:  "+ best_act[0] + " and best solver:  "+ best_solver[0]+"\n"
		print(msg)
		logs_file.write(msg)

		start = time.time()
		model = MLPClassifier(hidden_layer_sizes = (best_size[0],), activation = best_act[0], solver = best_solver[0])
		model.fit(x_train, y_train)
		pred = model.predict(x_val)
		end = time.time()

		msg = "[Finish] MLP validation. Results:\n Accuracy for validation set: " + str(accuracy_score(y_val, pred)) + ", confusion matrix for validation set: rows - trues, columns - predicted\n" 
		if dataset == 'fruit':
			cm = confusion_matrix(le.inverse_transform(y_val), le.inverse_transform(pred), labels= le.classes_)
			plot_confusion_matrix(cm, classes=le.classes_, title='Confusion matrix for fruit dataset with MLP', filename='../../cm_'+dataset+'_mlp', figsize = 7, fontsize = 6 )
			msg = msg + str(le.classes_) + "\n" + str(cm)
		else:
			cm = confusion_matrix(y_val, pred, [0, 1])
			plot_confusion_matrix(cm, classes={0,1}, title='Confusion matrix for car dataset with MLP', filename='../../cm_'+dataset+'_mlp' )
			msg = msg + "classes: no car - car" + str(cm)


		msg = msg + "\nPrecision for each class:\n" + str(precision_score(y_val, pred, average=None)) + "\nRecall for each class:\n" + str(recall_score(y_val, pred, average=None))

		print("[Finish] MLP validation.")
		logs_file.write(msg + "\n Elapsed time (validation): " + str(end-start) + "\n\n")

	# RANDOM FOREST --------------------
	if "randomForest" in algo_list:


		logs_file.write("[Start] Random Forest Parameter tunning \n")
		print("Random Forest Parameter tunning...")
		training_time = []
		testing_time = []
		# parameter selection with grid search for the size
		best_size = (0, 0)
		for size in classifier_feature_extraction_config.rf_size:
			acc = []
			for (train_index, test_index) in splits:
				x_train, x_test = data[train_index], data[test_index]
				y_train, y_test = target[train_index], target[test_index]
				start_tr = time.time()
				model = RandomForestClassifier(n_estimators=size)
				model.fit(x_train, y_train)
				end_tr = time.time()
				training_time.append(end_tr - start_tr)

				mean_acc = model.score(x_test, y_test)
				acc.append(mean_acc)
				end_test = time.time()
				testing_time.append(end_test - end_tr)

			mean_acc = np.mean(acc)
			logs_file.write("Mean accuracy for size="+str(size) + ":   " + str(mean_acc) + " with std " + str(np.std(acc)) +"\n")
			if mean_acc > best_size[1]:
				best_size = (size, mean_acc)
			
	
		logs_file.write("[Finish] Random Forest Parameter tunning \n")
		logs_file.write("Training times mean: " + str(np.mean(training_time)) + "\n" ) 
		logs_file.write("Testing times mean: " + str(np.mean(testing_time)) + "\n" ) 

		# validation
		logs_file.write("[Start] Random Forest Validation\n")
		msg = "Taking:\n Best size:  "+ str(best_size[0]) + "\n"
		print(msg)
		logs_file.write(msg)
	
		start = time.time()
		model =  RandomForestClassifier(n_estimators=best_size[0])
		model.fit(x_train, y_train)
		pred = model.predict(x_val)
		end = time.time()

		msg = "[Finish] RF validation. Results:\n Accuracy for validation set: " + str(accuracy_score(y_val, pred)) + "confusion matrix for validation set: rows - trues, columns - predicted\n" 
		if dataset == 'fruit':
			cm = confusion_matrix(le.inverse_transform(y_val), le.inverse_transform(pred), labels= le.classes_)
			plot_confusion_matrix(cm, classes=le.classes_, title='Confusion matrix for fruit dataset with random forest', filename='../../cm_'+dataset+'_rf', figsize = 7, fontsize = 6)
			msg = msg + str(le.classes_) + "\n" + str(cm)
		else:
			cm = confusion_matrix(y_val, pred, [0, 1])
			plot_confusion_matrix(cm, classes={0,1}, title='Confusion matrix for car dataset with random forest', filename='../../cm_'+dataset+'_rf' )
			msg = msg + "classes: no car - car" + str(cm)

		msg = msg + "\nprecision for each class:\n" + str(precision_score(y_val, pred, average=None)) + "\nrecall for each class:\n" + str(recall_score(y_val, pred, average=None))	

	

		print("[Finish] RF validation.")
		logs_file.write(msg + "\n Elapsed time (validation): " + str(end-start) + "\n\n")


	# K-NN -------------------------
	if "knn" in algo_list:

	
		logs_file.write("[Start] Knn Parameter tunning \n")
		print("Knn Parameter tunning...")
		training_time = []
		testing_time = []
		# parameter selection with grid search for k
		best_k = (0, 0)
		for k in classifier_feature_extraction_config.knn_k:
			acc = []
			for (train_index, test_index) in splits:
				x_train, x_test = data[train_index], data[test_index]
				y_train, y_test = target[train_index], target[test_index]
				start_tr = time.time()
				model = KNeighborsClassifier(n_neighbors=k)
				model.fit(x_train, y_train)
				end_tr = time.time()
				training_time.append(end_tr - start_tr)

				mean_acc = model.score(x_test, y_test)
				acc.append(mean_acc)
				end_test = time.time()
				testing_time.append(end_test - end_tr)

			mean_acc = np.mean(acc)
			logs_file.write("Mean accuracy for k="+str(k) + ": " + str(mean_acc) + " with std " + str(np.std(acc)) +"\n")
			if mean_acc > best_k[1]:
				best_k = (k, mean_acc)
			
		# parameter selection for weights
		best_weight = ('', 0)
		w_grid = ['uniform', 'distance']
		for weight in w_grid:
			acc = []
			for (train_index, test_index) in splits:
				x_train, x_test = data[train_index], data[test_index]
				y_train, y_test = target[train_index], target[test_index]
				start_tr = time.time()
				model = KNeighborsClassifier(weights = weight)
				model.fit(x_train, y_train)
				end_tr = time.time()
				training_time.append(end_tr - start_tr)

				mean_acc = model.score(x_test, y_test)
				acc.append(mean_acc)
				end_test = time.time()
				testing_time.append(end_test - end_tr)

			mean_acc = np.mean(acc)
			logs_file.write("Mean ccuracy for weighting method="+weight + ": " + str(mean_acc) + " with std " + str(np.std(acc)) +"\n")
			if mean_acc > best_weight[1]:
				best_weight = (weight, mean_acc)
	
		logs_file.write("[Finish] Knn Parameter tunning \n")
		logs_file.write("Training times mean: " + str(np.mean(training_time)) + "\n" ) 
		logs_file.write("Testing times mean: " + str(np.mean(testing_time)) + "\n" ) 

		# validation
		logs_file.write("[Start] Knn Validation\n")
		msg = "Taking:\n best k:  "+ str(best_k[0]) + " and best weight:  "+ str(best_weight[0]) + "\n"

		start = time.time()
		model = KNeighborsClassifier(n_neighbors=best_k[0], weights = best_weight[0])
		model.fit(x_train, y_train)
		pred = model.predict(x_val)
		end = time.time()

		msg = "[Finish] Knn validation. Results:\n" + "Accuracy for validation set: " + str(accuracy_score(y_val, pred)) +  "confusion matrix for validation set: rows - trues, columns - predicted\n" 

		if dataset == 'fruit':
			cm = confusion_matrix(le.inverse_transform(y_val), le.inverse_transform(pred), labels= le.classes_)
			plot_confusion_matrix(cm, classes=le.classes_, title='Confusion matrix for fruit dataset with Knn', filename='cm_'+dataset+'_knn', figsize = 7, fontsize = 6 )
			msg = msg + str(le.classes_) + "\n" + str(cm)
		else:
			cm = confusion_matrix(y_val, pred, [0, 1])
			plot_confusion_matrix(cm, classes={0,1}, title='Confusion matrix for car dataset with Knn', filename='cm_'+dataset+'_knn' )
			msg = msg + "classes: no car - car" + str(cm)

		msg = msg + "\nprecision for each class:\n" + str(precision_score(y_val, pred, average=None)) + "\nrecall for each class:\n" + str(recall_score(y_val, pred, average=None))

		print("[Finish] Knn validation.")
		logs_file.write(msg + "\n Elapsed time (validation): " + str(end-start) + "\n\n")



	logs_file.close()









