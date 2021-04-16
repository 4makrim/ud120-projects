#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]

data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### List outliers
for name, value in data_dict.items():
    deets = value
    for k, v in deets.items():
        # if k == 'salary':
        #     if float(v) > 1000000:
        #         print (name)
        if k == 'bonus':
            if float(v) > 5000000:
                print(name)

### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


