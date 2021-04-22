# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 22:20:41 2021

@author: maya_
"""
def evaluatePOIidentifier():
    import pickle
    import sys
    sys.path.append("../tools/")
    from feature_format import featureFormat, targetFeatureSplit
    
    data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )
    
    ### first element is our labels, any added elements are predictor
    ### features. Keep this the same for the mini-project, but you'll
    ### have a different feature list when you do the final project.
    features_list = ["poi", "salary"]
    
    data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
    # data = featureFormat(data_dict, features_list, sort_keys = True)
    
    labels, features = targetFeatureSplit(data)
    
    
    
    ### it's all yours from here forward!  
    
    ### Decision Tree
    from time import time
    from sklearn import tree
    ### Using min_samples_split = 2 accuracy = 90.8%
    ### Using min_samples_split = 50 accuracy = 91.2%
    #clf = tree.DecisionTreeClassifier(min_samples_split=40)
    clf = tree.DecisionTreeClassifier()
    t0 = time()
    clf.fit(features, labels)
    print("training time for all data:", round(time()-t0, 3) , "s")
    
    ### print accuracy
    print ("all data accuracy: ", clf.score(features, labels))
    
    
    # from email_preprocess import preprocess
    from classifyDT import classify
    
    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
        
        
    ### features_train and features_test are the features for the training
    ### and testing datasets, respectively
    ### labels_train and labels_test are the corresponding item labels
    # features_train, features_test, labels_train, labels_test = preprocess()
    
    clf = classify(features_train, labels_train, features_test, labels_test)
    ### expected result was 0.724
    print("#Features in data: ", len(features_train[0]))
    
