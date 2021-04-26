#!/usr/bin/python
import numpy as np
import sys
import pickle
sys.path.append("../tools/")

from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Explore enron data
from explore_enron_data import explore_enron_data
explore_enron_data()

def getMetrics(clf, features_test, labels_test):
    try:
        ### Make the predictions. 
        pred = clf.predict(features_test)
        print(pred)
        from sklearn.metrics import precision_score, recall_score, f1_score
        ### calculate f1_score
        print ("**F1 Score: ", f1_score(labels_test, pred, labels=np.unique(pred)))
        ### precision and recall 
        print("**Precision score: ", precision_score(labels_test, pred))
        print("**Recall score: ", recall_score(labels_test, pred))
    except: 
        print("There were no predictions for your feature")
        
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'exercised_stock_options', 'bonus', 
                  'total_payments'] 


### Load the dictionary containing the dataset
# with open("final_project_dataset.pkl", "rb") as data_file:
#     data_dict = pickle.load(data_file)
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", 
                             "rb") )
### Task 2: Remove outliers
### There are three outliers that can be removed. Found these by using the 
### enron61702insiderpay.pdf.

data_dict.pop("TOTAL", 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('YEAP SOON', 0)

### Add a couple features to this list. 
from newDataPoint import createNewPoints
createNewPoints(data_dict)
    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list = ['poi','salary', 'bonus', 'exercised_stock_options',  
                 'total_payments']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

        
### Why does this line not work here?
# features_list = ['poi','salary', 'exercised_stock_options', 'bonus', 
#                   'total_payments']
# data = featureFormat(my_dataset, features_list, 
#               sort_keys = '../tools/python2_lesson14_keys.pkl')
# labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Simple split of data into training and testing sets. 
# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### *************Stratified Data Split ***************************
# cv = StratifiedShuffleSplit(n_splits=1000, random_state = 42)

# for train_idx, test_idx in cv.split(features, labels): 
#     features_train = []
#     features_test  = []
#     labels_train   = []
#     labels_test    = []
#     for ii in train_idx:
#         features_train.append( features[ii] )
#         labels_train.append( labels[ii] )
#     for jj in test_idx:
#         features_test.append( features[jj] )
#         labels_test.append( labels[jj] )

### Perforning a fit with Naive Bayes algorithm
from classifyNB import classifyNB
clf = classifyNB(features_train, labels_train, features_test, 
                       labels_test)

### Performing a fit with AdaBoost.
from classifyAdaBoost import classifyAdaBoost
clf = classifyAdaBoost(features_train, labels_train, features_test, 
                       labels_test)

### One last algorithm to test - SVM 
### SVM seems to have the best accuracy at 0.689. We will use this algorithm.
from classifySVM import classifySVM
clf = classifySVM(features_train, labels_train, 
                                    features_test, labels_test)
# getMetrics(clf, features_test, labels_test)

#########################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")

t0 = time()
param_grid = {
         'C': [300,400],
          'gamma': [10],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(features_train, labels_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

### Even though SVM seemed to be the best in terms of accuracy, it was not 
### able to predict data points. The precision and recall for SVM were 0.
###########################################################################

### GridSearchCV using a stratified shuffle split per feedback suggestion.
### 4/26/2021 - It quite did not work. it fails in the fit. 
### Since my code above works and this was just a suggestion, 
### I have commented this out and will look at it later.

# print(" ********* Stratified Shuffle Split ********")
# # 1000 folds are used to make it as similar as possible to tester.py.
# folds = 50

# # We then store the split instance into cv and use it in our GridSearchCV.
# from sklearn.model_selection import StratifiedShuffleSplit
# cv = StratifiedShuffleSplit(
#      labels, test_size=5, random_state=42)
# grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), 
#                     param_grid, cv = cv, scoring='f1')
# grid.fit(features_train, labels_train)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

##########################################################################

### ****************** Importance of feature scaling *****************
from pca_fit import pca_fit
pca_fit(features_train, features_test, labels_train, labels_test)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
### Evaluate using a regular decision tree - this seems to be the best fit.
print("**** Final Classifier Algorithm Chosen: Regular Decision Tree *****")
from evaluate_poi_identifier import evaluatePOIidentifier
clf = evaluatePOIidentifier()


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

