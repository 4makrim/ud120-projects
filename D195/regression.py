# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:29:57 2021

@author: maya_
"""
import numpy as np
import pandas as pd
import sys
# sys.path.append("../tools/")

from readdata import read_data
alldata = read_data('flightdelays-2010-2020.csv')
### 

['year', ' month', 'carrier', 'carrier_name', 'airport', 'airport_name',
       'arr_flights', 'arr_del15', 'carrier_ct', ' weather_ct', 'nas_ct',
       'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted',
       ' arr_delay', ' carrier_delay', 'weather_delay', 'nas_delay',
       'security_delay', 'late_aircraft_delay', 'Unnamed: 21']

features_list = ['arr_del15', 'carrier_ct', ' weather_ct', 'nas_ct',
       'security_ct', 'late_aircraft_ct']
data = pd.DataFrame(alldata, columns=features_list)
features = data.keys()
target = []


data = data.dropna()


### Create a prediction target for late flights because of carrier delays.
for e in data['carrier_ct']:
    if(e > 0):
        target.append(1)
    else:
        target.append(0)


# from sklearn import svm
# from sklearn.model_selection import GridSearchCV

# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)
# clf.fit(data, target)
# GridSearchCV(estimator=svm.SVC(),
#               param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
# sorted(clf.cv_results_.keys())


### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(data, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

# import sys
# sys.path.append("../final_project/")
from pca_fit import pca_fit
pca_fit(feature_train, feature_test, target_train, target_test)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, target_train)

print("Carrier Delay: ", reg.predict(feature_train))
print("slope:", reg.coef_)
print("intercept: ", reg.intercept_)

print ("\n##### stats on test dataset #######\n")
print("r-squared score:", reg.score(feature_test, target_test))

print ("\n##### stats on training dataset #######\n")
print("r-squared score:", reg.score(feature_train, target_train))


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

## labels for the legend
plt.scatter("Arrival Delay > 15 minutes", "Carrier Count", color=test_color, label="test")
plt.scatter("Arrival Delay > 15 minutes", "Carrier Count", color=train_color, label="train")




## draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass

reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()


# def evaluate_metrics(features, labels):
#     """ Perform a GridSearchCV to determine the best combination of 
#     parameters for the SVM algorithm that acheives the highest level
#     of precision and recall. 
#     """
    
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.svm import SVC
#     parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#     svr = SVC()
#     ### n_jobs=-1 - use all processors
    
#     clf = GridSearchCV(svr, parameters, n_jobs=-1, verbose=3 )
#     clf.fit(features, labels)
#     print(sorted(clf.cv_results_.keys()))
#     print ("Ideal parameters: ", clf.best_params_)


# evaluate_metrics(features, labels)