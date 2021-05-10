# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:25:43 2021

@author: maya_
"""

from time import time

def classifyDT(features_train, labels_train, features_test, labels_test):
    """
    returns a decision tree classifier with the given training and test datasets.

    """
    ### your code goes here--should return a trained decision tree classifer
    from sklearn import tree
    ### Using min_samples_split = 2 accuracy = 90.8%
    ### Using min_samples_split = 50 accuracy = 91.2%
    #clf = tree.DecisionTreeClassifier(min_samples_split=40)
    clf = tree.DecisionTreeClassifier()
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3) , "s")
    
    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t1, 3) , "s")
    
    ### count number of positive predictions
    count = 0
    for i in pred:
        if i == 1:
            count += 1
    print("no. postive predictions: ", count)
    #print("classifyDT:", pred)
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
    print ("F1 Score: ", round(f1_score(labels_test, pred, labels=np.unique(pred)),3))
    print("Precision score: ", round(precision_score(labels_test, pred), 3))
    print("Recall score: ", round(recall_score(labels_test, pred), 3))
    
    ### print accuracy
    # from sklearn.metrics import accuracy_score
    # print("Accuracy: ", accuracy_score(pred, labels_test))
    
    ### Alternate method
    print ("Decision Tree Classifier Accuracy: ", round(clf.score(features_test, labels_test),3))
    #########################################################
    
    
    return clf