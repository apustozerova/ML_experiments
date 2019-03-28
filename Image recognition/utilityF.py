
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import itertools
from matplotlib.pyplot import figure
import numpy as np

# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, filename='cm', figsize = 3, fontsize = 8):
	"""
	This function plots the confusion matrix.
	"""
	figure(figsize=(figsize, figsize))
	img = plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title, fontsize=fontsize)
	cbar = plt.colorbar(img,fraction=0.046, pad=0.04)
	cbar.ax.tick_params(labelsize=fontsize) 


	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=60, fontsize=fontsize)
	plt.yticks(tick_marks, classes, fontsize=fontsize)

	fmt = 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=6)

	plt.ylabel('True label', fontsize=fontsize)
	plt.xlabel('Predicted label', fontsize=fontsize)
	plt.tight_layout()
	plt.savefig(filename + ".png")
	plt.close()




def log_model_summary(log_file, model):
	# Log model summary
	stringlist = []
	model.summary(print_fn=lambda x: stringlist.append(x))
	short_model_summary = "\n".join(stringlist)
	log_file.write("Model summary:\n" + str(short_model_summary) +"\n")


def plot_history(history, epochs, filename):
	colors = {'loss':'r', 'acc':'b', 'val_loss':'m', 'val_acc':'g'}
	plt.figure(figsize=(10,6))
	plt.title("Training Curve") 
	plt.xlabel("Epoch")

	for measure in history.keys():
	    color = colors[measure]
	    plt.plot(range(1,epochs+1), history[measure], color + '-', label=measure)  # use last 2 values to draw line

	plt.legend(loc='upper left', scatterpoints = 1, frameon=False)
	plt.savefig(filename)



