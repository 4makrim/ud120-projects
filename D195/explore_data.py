# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:17:00 2021

@author: maya_
"""

#!/usr/bin/python

""" 
    Starter code for exploring the dataset, loads up the dataset.

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from readdata import read_data

# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
# The following code reads all the Gapminder data into Pandas DataFrames. 

"""
Read the data files and prepare the data for calculations and graphs
"""
fd = read_data('flightdelays-2010-2020.csv')

# fd.head()

fd.keys()
count = 0

### Code below determines how if we need to do additional data wrangling.
### Total number of records
print(len(fd))
### Total number of keys
print (fd.keys())

import builtins
### Total number of non-null values 
for key in fd.keys():
    peds = fd[key]
    count = 0
    if (type(peds[0]) == type("str")):
         count = builtins.sum(1 for e in peds if e != "")
    else:
         count =builtins.sum(1 for e in peds if e >= 0)

    print (f'{key:20} : {count:5}')
               



