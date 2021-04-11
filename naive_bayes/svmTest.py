#!/usr/bin/python

""" Use SVM to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson 3 video, and make a plot that
    visually shows the decision boundary """

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

#from ClassifyNB import classify
from printImg import showImage


features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
# clf = SVC(kernel="linear")

### The rbf gives a squiggly line. 
### The accuracy is 0.948 for gamma=10, C=400
clf = SVC(kernel="rbf", gamma=10, C=400)

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("Accuracy: ", acc)

#### store your predictions in a list named pred

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.
#clf = classify(features_train, labels_train, features_test, labels_test)



### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test, "svmtest.png")
#output_image("test.png", "png", open("test.png", "rb").read())

### cannot open image file using the starter file command above.
showImage("svmtest.png")




