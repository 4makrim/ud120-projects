# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:32:06 2021

@author: maya_
"""

import pandas as pd

# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
# The following code reads all the Gapminder data into Pandas DataFrames. 

"""
Read the data files and prepare the data for calculations and graphs
"""
def read_data(csvfile):
    path = './'
    datafile = pd.read_csv(path + csvfile)
    return datafile
