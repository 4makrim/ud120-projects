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
    
    ### Add a couple features to this list. 
    from newDataPoint import createNewPoints
    createNewPoints(data_dict)
    
    ### first element is our labels, any added elements are predictor
    ### features. Keep this the same for the mini-project, but you'll
    ### have a different feature list when you do the final project.
    #features_list = ['poi','salary', 'exercised_stock_options', 'bonus', 
    #                    'total_payments']
    features_list = ['poi', 
                      'bonus',
                        'deferral_payments',
                        'deferred_income',
                        'director_fees',
                      'exercised_stock_options',
                        'expenses',
                        'fraction_from_poi',   ### new feature
                        'fraction_to_poi',     ### new feature
                        'loan_advances',
                        'long_term_incentive',           
                        'restricted_stock', 
                        'restricted_stock_deferred',
                      'salary',
                      'total_payments', 
                        'total_stock_value'                      
                      ]
    # features_list = ['poi',
    #                   'salary',
    #                   'bonus',
    #                   'exercised_stock_options',
    #                   'total_payments']
    data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
    #data = featureFormat(data_dict, features_list, sort_keys = True)
    ### Why does the sort_keys have to be True?
        
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
    print ("Decision Tree Accuracy on All the data: ", round(clf.score(features, labels), 3))
    
    
    # from email_preprocess import preprocess
    from classifyDT import classifyDT
    
    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
        
        
    ### features_train and features_test are the features for the training
    ### and testing datasets, respectively
    ### labels_train and labels_test are the corresponding item labels
    # features_train, features_test, labels_train, labels_test = preprocess()
    
    clf = classifyDT(features_train, labels_train, features_test, labels_test)
    ### expected result was 0.724
    print("no. features used in classification: ", len(features_train[0]))
    
    ### Determine the importance of the features that we chose. 
    ### salary and bonus seem to be the highest.
    lat = [i for i in clf.feature_importances_]
    ### use the cut off 0.2 to determine the features of importance.
    def condition(x): return x > 0.2
    #def condition(x): return x
    output = [idx for idx, element in enumerate(lat) if condition(element)]
    print("output:", output)
    for i in output:
        print("importance: of ", features_list[i], " is ",round(lat[i],3))
        
    #getMetrics(clf, features_test, labels_test)
    ### above line is showing 1.0 for the metrics. This may have something to do 
    ### with the sort_keys not working in this file.
    
    return clf
    

def main():
    evaluatePOIidentifier()
    
if __name__ == '__main__':
    main()