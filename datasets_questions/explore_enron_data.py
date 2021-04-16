#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", 
                                  "rb"))
count = 0
sal_count = 0
email_count = 0
total_payout_count = 0
poi_nopayment_count = 0

for name, value in enron_data.items():


    peds = value
    for key, value in peds.items():
        if key == "poi":
            if value == 1:
                count += 1
                print("**POI**: ", name)
                if key == 'total_payments':
                    if math.isnan(float(value)):
                        poi_nopayment_count += 1
                
        elif key == "salary":
            if math.isnan(float(value)):
                # do nothing
                sal_count += 0
            else:
                sal_count += 1
        elif key == 'email_address':
            if value  and value != 'NaN' and not value.isspace():
                print (name, " ", value)
                email_count += 1
        elif key == 'total_payments':
            if math.isnan(float(value)):
                # do nothing
                total_payout_count += 1

                
                
print("===========\nTotal: ", count, "\n===========")
print("===========\nSalary Count: ", sal_count, "\n===========")
print("===========\nEmail Count: ", email_count, "\n===========")
print(total_payout_count/146)
print("Percentage of POIs that have no total payments: ", poi_nopayment_count/count)
