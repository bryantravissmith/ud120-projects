#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
max = 0
name = 0
count = 0

def count_value(key):
    count = 0
    total = 0
    for person in enron_data:
        if person != "TOTAL":
            total += 1
            value = enron_data[person][key]
            if value != "NaN":
                count += 1

    return count, total

def count_value_poi(key):
    count = 0
    total = 0
    for person in enron_data:
        if person != "TOTAL" and enron_data[person]['poi']:
            total += 1
            value = enron_data[person][key]
            if value != "NaN":
                count += 1

    return count, total
#print count_value('poi')
#print count_value('email_address')
print count_value('salary')
print count_value('total_payments')

