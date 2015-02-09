#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print len(features_train[0])


clf = DecisionTreeClassifier(min_samples_split=40)
t1 = time()
clf.fit(features_train,labels_train)
print "Training:", round(time()-t1,3)
t2 = time()
pred = clf.predict(features_test)
print "Predicting:", round(time()-t2,3)
from sklearn.metrics import accuracy_score

print accuracy_score(pred,labels_test)


#########################################################
### your code goes here ###


#########################################################

