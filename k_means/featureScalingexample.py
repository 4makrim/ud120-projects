# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 00:21:21 2021

@author: maya_
"""
""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    
    maxi = max(arr)
    mini = min(arr)
    for i in arr:
        if not i == maxi and not i == mini:
            return (i - mini)/(maxi - mini)
    return None

# tests of your feature scaler--line below is input data
data = [115, 160, 175, 165]
print ("Feature Scaling: ", featureScaling(data))


from sklearn.preprocessing import MinMaxScaler
import numpy as np

### Katie says in her video 10.11 that we need to use floats
### This code works as is.
weights = np.array([[115],[160],[175],[165]])
scalar = MinMaxScaler()
rescaled_weight = scalar.fit_transform(weights)
print(rescaled_weight)