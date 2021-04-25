# -*- coding: utf-8 -*-

from time import time

def classifySVM(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your SVM Decision Tree classifier """
    ########################## SVM #################################
    ### we handle the import statement and SVC creation for you here
    from sklearn.svm import SVC
    # clf = SVC(kernel="linear")
    
    print("*** Classifier SVM Decision Tree ***")
    ### The rbf gives a squiggly line. 
    ### The accuracy is 0.948 for gamma=10, C=400
    #clf = SVC(kernel="rbf", gamma=10, C=400)
    ### The accuracy is 0.952 for kernel='rbf', gamma=10, C=300
    clf = SVC(kernel='rbf', gamma=10, C=300)

    ### Linear fit does not work - takes too long. 
    #clf = SVC(kernel='linear', gamma=10, C=300)
    #### now your job is to fit the classifier
    #### using the training features/labels, and to
    #### make a set of predictions on the test data
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3) , "s")
    
    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t1, 3) , "s")
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    print("SVM Accuracy: ", acc)
    return clf