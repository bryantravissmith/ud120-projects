#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = SVC(kernel="rbf",C=10000)
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

print pred[10], pred[26], pred[50]
from sklearn.metrics import accuracy_score
print accuracy_score(pred,labels_test)

"""
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel="linear")
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"
from sklearn.metrics import accuracy_score
print accuracy_score(pred,labels_test)

clf = SVC(kernel="rbf")
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"
from sklearn.metrics import accuracy_score
print accuracy_score(pred,labels_test)

import math
for i in range(4,5):
    clf = SVC(kernel="rbf",C=math.pow(10,i))
    t0 = time()
    clf.fit(features_train,labels_train)
    print "training time:", round(time()-t0, 3), "s"
    t1 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time()-t1, 3), "s"
    from sklearn.metrics import accuracy_score
    print accuracy_score(pred,labels_test)
#########################################################
### your code goes here ###

#########################################################

"""
