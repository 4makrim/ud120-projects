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
def explore_enron_data():
    import pickle
    import math
    enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", 
                                      "rb"))
    count = 0
    sal_count = 0
    email_count = 0
    total_payout_count = 0
    poi_nopayment_count = 0
    poi_payment_count = 0
    
    for name, value in enron_data.items():
    
    
        peds = value
        if (peds['poi'] == 1):
            count += 1
            if (math.isnan(float(peds['total_payments']))):
                poi_nopayment_count += 1
            else:
                poi_payment_count += 1
            
        #print("peds: ", peds['total_payments'])
        for key, value in peds.items():
                    
            if key == "salary":
                if math.isnan(float(value)):
                    # do nothing
                    sal_count += 0
                else:
                    sal_count += 1
            elif key == 'email_address':
                if value  and value != 'NaN' and not value.isspace():
                    #print (name, " ", value)
                    email_count += 1
            elif key == 'total_payments':
                if math.isnan(float(value)):
                    # do nothing
                    total_payout_count += 1
    
                    
    print("Total no. of individual accounts: ", len(enron_data))                
    print("No. of valid salary entries: ", sal_count)
    print("No. of valid email IDs we have data for: ", email_count)
    print("How many people received a total payout? : ", total_payout_count)
    print("Total no. of POI: ", count)
    print("Percentage of people who received payout: ", 
         ( total_payout_count/len(enron_data))*100)
    
    ### Every one of the POIs received total payments of some amount. 
    ### confirmed by the result of the following line.
    print("Percentage of POIs that have received total payments: ", 
              (poi_payment_count/count)*100)
    print("Percentage of POIs that have no total payments: ", 
              (poi_nopayment_count/count)*100)

