#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from time import time

max_accuracy = 0
max_neighbors = 0
max_weight = 'none'
max_metric = 'none'

weights = ['uniform','distance']
metrics = ['euclidean', 'manhattan', 'chebyshev', 'canberra']
for i in range(2,100):
    for weight in weights:
        for metric in metrics:
            clf = KNeighborsClassifier(i,weights=weight,metric=metric)
            t = time()
            clf.fit(features_train, labels_train)
            #print "Training Time:",round(time()-t,3)
            t = time()
            pred = clf.predict(features_test)
            #print "Predict Time:", round(time()-t,3)
            accuracy = accuracy_score(pred,labels_test)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_neighbors = i
                max_metric = metric
                max_weight = weight

print max_neighbors, max_accuracy, max_metric, max_weight

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

max_accuracy = 0
max_estimator = 0
max_learning_rate = 0

estimators = range(10,210,10)
learning_rates = [0.01,0.1,1,10]

for estimator in estimators:
    for learning_rate in learning_rates:
        clf = AdaBoostClassifier(n_estimators=estimator,learning_rate=learning_rate)
        t = time()
        clf.fit(features_train, labels_train)
        print "Training Time:",round(time()-t,3)
        t = time()
        pred = clf.predict(features_test)
        print "Predict Time:", round(time()-t,3)
        accuracy = accuracy_score(pred,labels_test)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_estimator = estimator
            max_learning_rate = learning_rate

print max_accuracy, max_estimator, max_learning_rate


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print accuracy

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    print "name error"
    pass
