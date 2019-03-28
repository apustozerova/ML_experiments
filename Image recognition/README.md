# Group19: Third assignment for Machine Learning course

This folder includes the python implementation for the third assignment of ML course.

## Prerequisites

Our experiments were tested with Python 3.7.2. The following packages are required:
- Numpy 1.16.1, 
- Scikit-learn 0.20.2, 
- Matplotlib 3.0.2, 
- Pillow 5.4.1, 
- Theano 1.0.4, 
- Keras 2.2.4,
- python-resize-image.

Additionally it is required a manually compiled version of OpenCV 4.0.0 which includes the non-free package SIFT.

## Running

To run our experiments call the python script *group19.py* with desired arguments.

This script can be configured with the following arguments:
- **-p/--path**: Path to the images (example: *-p CarData/CarData*)
- **-a/--algorithm**: Learning algorithms to try. Possible choices: 
*dl*: CNN and fully connected networks,
*mlp*: Multilayer perceptron,
*randomForest*: Random forest,
*knn*: K-nn.
- **-d/--dataset**: Dataset to learn. Possible values: *car* and *fruit*.
- **-b/--bins**: Number of bins for the colour histogram.

Examples of script calls:
```
python3 group19.py -p FIDS30/FIDS30/ -a knn -a dl -d fruit
```

```
python3 group19.py -p CarData/CarData/ -a randomForest -a mlp -a dl -d car

```

## Further Configuration

There are 3 python files that allow to configure parameters of experiments.

### Traditional classification

The file *classifier_feature_extraction_config.py* allows to configure parameters to be tested for the traditional classifiers.


### Deep Learning

Files *car_nn_config.py* and *fruit_cnn_config.py* allow to configure parameters for convolution neural networks and simple connected neural networks for both
car and fruit dataset, respectively.


## Datasets


Full datasets were too big to include in the final zip for submission.

Please download the datasets in the following links:
- Car: http://cogcomp.org/Data/Car/
- Fruits: http://www.vicos.si/Downloads/FIDS30


