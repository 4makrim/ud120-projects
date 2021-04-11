# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:43:21 2021

@author: maya_
"""
from time import time

def classify(features_train, labels_train, features_test, labels_test):
    
    ### your code goes here--should return a trained decision tree classifer
    from sklearn import tree
    ### Using min_samples_split = 2 accuracy = 90.8%
    ### Using min_samples_split = 50 accuracy = 91.2%
    clf = tree.DecisionTreeClassifier(min_samples_split=40)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3) , "s")
    
    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t1, 3) , "s")
    
    ### print accuracy
    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred, labels_test))
    
    ### Alternate method
    print (clf.score(features_test, labels_test))
    #########################################################
    
    
    return clf