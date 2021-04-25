# -*- coding: utf-8 -*-
from time import time

def classifyAdaBoost(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your AdaBoost classifier """
    ### visualization code (prettyPicture) to show you the decision boundary
    from sklearn.ensemble import AdaBoostClassifier
    
    ### Seems like 50 is the lowest ideal value for 0.924 accuracy. 
    estimator = 13
    
    print("*** Classifier AdaBoost ***")
    print("Estimator=", estimator)
    clf = AdaBoostClassifier(n_estimators=estimator, random_state=100)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3) , "s")
    
    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t1, 3) , "s")
    
    ### print accuracy
    from sklearn.metrics import accuracy_score
    print("AdaBoost Accuracy: ", accuracy_score(pred, labels_test))
    return clf
