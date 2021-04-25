#!/usr/bin/python
import numpy as np
import sys
import pickle
sys.path.append("../tools/")

from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

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
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )
### Task 2: Remove outliers
### There is one outlier that can be removed. TOTAL
data_dict.pop("TOTAL", 0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list = ['poi','salary', 'exercised_stock_options', 'bonus', 'total_payments']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Why does this line not work here?
# features_list = ['poi','salary', 'exercised_stock_options', 'bonus', 'total_payments']
# data = featureFormat(my_dataset, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
# labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

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


##########################################################################


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

### Determine the importance of the features that we chose. 
### salary and bonus seem to be the highest.
lat = [i for i in clf.feature_importances_]
def condition(x): return x > 0.2
output = [idx for idx, element in enumerate(lat) if condition(element)]
print("output:", output)
for i in output:
    print("importance:",lat[i])
    
#getMetrics(clf, features_test, labels_test)
### above line is showing 1.0 for the metrics. This may have something to do 
### with the sort_keys not working in this file.



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

