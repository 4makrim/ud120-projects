# -*- coding: utf-8 -*-

import pickle
### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
max_exercised_stock_options = 0
min_exercised_stock_options = 100000000   
    
def getMinMax(element):
    maxi = 0
    mini = 1000000000
    for name, value in data_dict.items():
        deets = value
        for k, v in deets.items():       
            if k == element:           
                if float(v) > maxi:
                    maxi = float(v)
                elif float(v) < mini:
                    mini = float(v)
    return mini, maxi

min_exercised_stock_options, max_exercised_stock_options = getMinMax('exercised_stock_options')
min_salary, max_salary = getMinMax('salary')

print("Minimum exercised stock options: ", min_exercised_stock_options)
print("Maximum exercised stock options: ", max_exercised_stock_options)
print("Minimum salary: ", min_salary)
print("Maximum salary: ", max_salary)