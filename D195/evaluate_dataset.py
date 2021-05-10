# -*- coding: utf-8 -*-
"""
Created by Maya May 9, 2021
"""
import pandas as pd
from readdata import read_data
alldata = read_data('flightdelays-2010-2020.csv')

def evaluateDataset():
    """
    Returns
    -------
    clf : Classifier Model
        Evaluate and choose a classifier with the best suited options.
        Evaluation includes accuracy, F1 score, precision and recall. 
        The output lists the importance of the selected features. 

    """
    ### Add a couple features to this list. 
    # from newDataPoint import createNewPoints
    # createNewPoints(data_dict)
    
    alldata = read_data('flightdelays-2010-2020.csv')

    key_features_list = ['arr_del15', 'carrier_ct', ' weather_ct', 'nas_ct',
       'security_ct', 'late_aircraft_ct']
    features_list = ['arr_del15', ' weather_ct', 'nas_ct',
       'security_ct', 'late_aircraft_ct']
    data = pd.DataFrame(alldata, columns=key_features_list)
    target = []

    data = data.dropna()

    ### Create a prediction target for late flights because of carrier delays.
    for e in data['carrier_ct']:
        if(e > 0):
            target.append(1)
        else:
            target.append(0)
        
    ### Remove carrier_ct from the list as we used that to create the 
    ### target data for the predictions. 
    ### carrier_ct = weather_ct + nas_ct + security_ct + late_aircraft_ct    
    newdata = pd.DataFrame(data, columns=features_list)    
    
    data = newdata
    
    ### Decision Tree
    from time import time
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    t0 = time()
    clf.fit(data, target)
    print("training time for all data:", round(time()-t0, 3) , "s")
    
    ### print accuracy
    print ("Decision Tree Accuracy on All the data: ", 
           round(clf.score(data, target), 3))
       
    from classifyDT import classifyDT
    

    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(data, target, test_size=0.5, random_state=42)        
        
    ### features_train and features_test are the features for the training
    ### and testing datasets, respectively
    ### labels_train and labels_test are the corresponding item labels
    # features_train, features_test, labels_train, labels_test = preprocess()
    
    clf = classifyDT(features_train, labels_train, features_test, labels_test)
    
    ### Determine the importance of the features that we chose. 
    lat = [i for i in clf.feature_importances_]
    ### use the cut off 0.2 to determine the features of importance.
    #def condition(x): return x > 0.2
    def condition(x): return x
    output = [idx for idx, element in enumerate(lat) if condition(element)]
    print("output:", output)
    for i in output:
        print("importance: of ", features_list[i], " is ",round(lat[i],3))
    
    return clf
    

def main():
    evaluateDataset()
    
if __name__ == '__main__':
    main()
