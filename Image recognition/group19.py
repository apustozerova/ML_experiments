#--------------------------READING PARAMETERS----------------------------------------------

import argparse
import os
import time

parser = argparse.ArgumentParser()
# here further options can be added in this format, optparse does reasonable things with it on its own :)
parser.add_argument("-p", "--path", required=True, dest="imagePath", help="Path to data set.")
parser.add_argument("-a", "--algorithm", action='append', dest="algo_list", default=[], choices=['dl','mlp','randomForest','knn'], help="Algorithm(s) to use. Possible values: mlp, randomForest, knn and dl. To use more than one classification algorithm use this argument multiple times, ex: -a mlp -a knn -a dl")
parser.add_argument("-b", "--bins", dest="bins", default="64", type=int, help="number of bins for the colour histogram")
parser.add_argument("-d", "--dataset", dest="dataset", default="fruit",  choices=['fruit'],help="Dataset to learn. Possible values: fruit.")

# parsing of arguments
options = parser.parse_args()
imagePath = os.path.abspath(options.imagePath)
algo_list = options.algo_list
n_bins = options.bins
dataset = options.dataset

import classifier_feature_extraction
import fruits_DL

if 'mlp' in algo_list or 'randomForest' in algo_list or 'knn' in algo_list:
	classifier_feature_extraction.main(imagePath, algo_list, dataset, n_bins)

if dataset == 'fruit' and 'dl' in algo_list:
	fruits_DL.main(imagePath)
