# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:27:16 2021

@author: maya_
"""

import math
import pickle
from get_data import getData

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if not math.isnan(float(all_messages)):
        fraction = poi_messages/all_messages


    return fraction


data_dict = getData() 

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    print
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    print (fraction_from_poi)
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    print (fraction_to_poi)
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
    
 
### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot
from feature_format import featureFormat
features = ["from_poi_to_this_person", "from_this_person_to_poi"]
data = featureFormat(submit_dict, features)
for point in data:
    from_poi_to_this_person = point[0]
    from_this_person_to_poi = point[1]
    matplotlib.pyplot.scatter( from_poi_to_this_person, from_this_person_to_poi )

matplotlib.pyplot.xlabel("from_poi_to_this_person")
matplotlib.pyplot.ylabel("from_this_person_to_poi")
matplotlib.pyplot.show()
# from feature_format import targetFeatureSplit
# features_list = ["from_poi_to_this_person", "from_this_person_to_poi"]
# #data = featureFormat( data_point, features_list, remove_any_zeroes=True)

# target, features = targetFeatureSplit( submit_dict)

# ### training-testing split needed in regression, just like classification
# from sklearn.model_selection import train_test_split
# feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
# train_color = "b"
# test_color = "r"

# import matplotlib.pyplot as plt
# for feature, target in zip(feature_test, target_test):
#     plt.scatter( feature, target, color=test_color ) 
# for feature, target in zip(feature_train, target_train):
#     plt.scatter( feature, target, color=train_color ) 

# ### labels for the legend
# plt.scatter(feature_test[0], target_test[0], color=test_color, label="A")
# plt.scatter(feature_test[0], target_test[0], color=train_color, label="B")


#####################

def submitDict():
    return submit_dict
