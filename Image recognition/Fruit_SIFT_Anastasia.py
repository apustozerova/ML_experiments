import cv2
import glob
import os
import numpy as np
import imghdr #to check format of the picture

#from PIL import Image

img_paths = []
target_labels = []

#finding all the pictres, storing lables and targets
for root, dirs, files in os.walk("FIDS30"):
    for file in files:
        if file.endswith(".jpg"):
            classes = root.replace("FIDS30/", "")
            target_labels.append(classes)
            img_paths.append(os.path.join(root, file))

#remembering the names
image_names = []
for ipath in img_paths:
    image_names.append(os.path.basename(ipath))

#SIFT
features_SIFT = []
k=0

img_paths1 = img_paths[:10]
for file in img_paths1:
    if imghdr.what(file)=='jpeg':
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        print(k)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None) # get SIFT key points and descriptors
       	features_SIFT.append(des)
        np.save("FruitSIFT/"+image_names[k],des) #saving description of keaypoints for each picture in the folder FruitSIFT with the original name 
    k+=1 #just to check the progress

