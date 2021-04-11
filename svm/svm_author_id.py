#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### Train with a smaller data set to check the difference in accuracy.
features_train = features_train[:int(len(features_train)/100)] 
labels_train = labels_train[:int(len(labels_train)/100)]

########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
# clf = SVC(kernel="linear")

clf = SVC(kernel="rbf", gamma=100, C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print("SVM training time:", round(time()-t0, 3) , "s")

t1 = time()
pred = clf.predict(features_test)
print("SVM prediction time:", round(time()-t1, 3) , "s")

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("Accuracy: ", acc)

#########################################################


