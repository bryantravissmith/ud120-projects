#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL",0)
data = featureFormat(data_dict, features)
for x in data_dict:
    salary = data_dict[x]['salary']
    bonus = data_dict[x]['bonus']
    if salary != "NaN" and salary > 1000000 and bonus != "NaN" and bonus >  5000000:
        print x, data_dict[x]['bonus']
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
### your code below



