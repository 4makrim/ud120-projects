# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:56:52 2021

@author: maya_
"""

#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.insert(1, '../naive_bayes')
import class_vis

from prep_terrain_data import makeTerrainData

# import matplotlib.pyplot as plt
# import numpy as np
# import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train, features_test, labels_test)







#### grader code, do not modify below this line

class_vis.prettyPicture(clf, features_test, labels_test, "test.png")
#output_image("test.png", "png", open("test.png", "rb").read())

# importing Image class from PIL package 
from PIL import Image   
# creating a object 
im = Image.open("test.png")   
im.show()
