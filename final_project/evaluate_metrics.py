# -*- coding: utf-8 -*-

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

features_list = ['poi','salary', 'exercised_stock_options', 'bonus', 'total_payments'] # You will need to use more features

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


def evaluate_metrics(features, labels):
    """ Perform a GridSearchCV to determine the best combination of 
    parameters for the SVM algorithm that acheives the highest level
    of precision and recall. 
    """
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    parameters = {'kernel':('linear', 'rbf'), 'C':[300, 400]}
    svr = SVC()
    ### n_jobs=-1 - use all processors
    
    clf = GridSearchCV(svr, parameters, n_jobs=-1, verbose=3 )
    clf.fit(features, labels)
    print(sorted(clf.cv_results_.keys()))
    print ("Ideal parameters: ", clf.best_params_)
    

evaluate_metrics(features, labels)