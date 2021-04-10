# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 06:01:03 2021

@author: maya_
"""
import sys
from time import time

# This functionwas was called NBAccuracy in starter code. 
def classify(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    ### create classifier
    clf = GaussianNB()
    
    ### fit the classifier on the training features and labels
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3) , "s")

    ### use the trained classifier to predict labels for the test features
    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t1, 3) , "s")

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = clf.score(features_test, labels_test)
    
    ### Why does it say return accuracy when the caller is expecting a clf?
    return clf


