# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 23:03:10 2021

@author: maya_
"""
### Using the scipy to calculate entropy for a feature with 2 similar 
### and 1 different value.
import scipy.stats
print (scipy.stats.entropy([2,1],base=2))