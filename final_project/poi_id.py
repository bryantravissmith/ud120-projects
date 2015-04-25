#!/usr/bin/python

import sys
#import os
import pickle
import numpy as np

sys.path.append("../tools/")
#sys.path.append(os.getcwd()+'/tools/')
#sys.path.append(os.getcwd()+'/final_project/')
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'bonus',
                 'restricted_stock',
                 'shared_receipt_with_poi',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 #'email_address',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi_ratio',
                 'from_poi_to_this_person_ratio']

finance = ['salary',
             'deferral_payments',
             'total_payments',
             'exercised_stock_options',
             'bonus',
             'restricted_stock',
             'restricted_stock_deferred',
             'total_stock_value',
             'expenses',
             'loan_advances',
             'other',
             'director_fees',
             'deferred_income',
             'long_term_incentive']


email_col = ['from_messages','to_messages','from_this_person_to_poi','from_poi_to_this_person',
             'from_this_person_to_poi_ratio','from_poi_to_this_person_ratio']

features_list = ['poi','from_this_person_to_poi_ratio','from_poi_to_this_person_ratio']
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#data_dict = pickle.load(open("final_project/final_project_dataset.pkl", "r") )



### Task 2: Remove outliers

import pandas as pd
data = pd.DataFrame.from_dict(data_dict)
del data['TOTAL']
del data['THE TRAVEL AGENCY IN THE PARK']

#Create New Variables and use replace NaN to 0.0
data2 = data.transpose()
data2 = data2.replace('NaN',0.0)
data2['from_this_person_to_poi_ratio'] = data2['from_this_person_to_poi']/data2['to_messages']
data2['from_poi_to_this_person_ratio'] = data2['from_poi_to_this_person']/data2['from_messages']
data2 = data2.replace('NaN',0.0)
data2['poi'] = data2.poi.astype('float')
data = data2.transpose()

#Reset the data dictionary with total and travel agency reviewed
data_dict = data.to_dict()

data4 = data2.copy()
data6 = data2.copy()

#normalize the financial data, make two copies, and remove outliers from one copy
for key in finance:
    data4[key] = (data2[key]-data2[key].mean())/data2[key].std()
    data6[key] = (data2[key]-data2[key].mean())/data2[key].std()
    data4 = data4[data4[key] < 3]
    data4 = data4[data4[key] > -3]


#remove none outliers from copy to get a list of outliers
data5 = data2.copy().transpose()
list = data4.transpose().columns.values.tolist()
for name in list:
    del data5[name]

outliers = data5.columns.values.tolist()
print outliers


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing



### Remove Outliers:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


features = ['salary',
             'to_messages',
             'deferral_payments',
             'total_payments',
             'exercised_stock_options',
             'bonus',
             'restricted_stock',
             'shared_receipt_with_poi',
             'restricted_stock_deferred',
             'total_stock_value',
             'expenses',
             'loan_advances',
             'from_messages',
             'other',
             'from_this_person_to_poi',
             'director_fees',
             'deferred_income',
             'long_term_incentive',
             'from_poi_to_this_person',
             'from_this_person_to_poi_ratio',
             'from_poi_to_this_person_ratio']

labels = data2['poi'].values
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score
from sklearn.cross_validation import train_test_split, KFold

print "       "
print "Exploring 2 Variable Results with 10-fold CV:"
print "       "

for i in range(len(features)):
    for j in range(i+1,len(features)):
                features_list = [features[i],features[j]]
                clf = DecisionTreeClassifier()
                nb = GaussianNB()
                temp = data2[features_list]
                total_predictions_tree = []
                total_predictions_nb = []
                total_labels = []
                kf = KFold(len(data2), n_folds=10,random_state=42)

                for train_index, test_index in kf:
                    X_train, X_test = temp.values[train_index], temp.values[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    clf.fit(X_train,y_train)
                    nb.fit(X_train,y_train)
                    total_predictions_tree = total_predictions_tree + clf.predict(X_test).tolist()
                    total_predictions_nb = total_predictions_nb + nb.predict(X_test).tolist()
                    total_labels = total_labels + y_test.tolist()

                r_score = recall_score(total_predictions_tree,total_labels)
                p_score = precision_score(total_predictions_tree,total_labels)
                nb_r_score = recall_score(total_predictions_nb,total_labels)
                nb_p_score = precision_score(total_predictions_nb,total_labels)
                if nb_r_score > 0.35 and nb_p_score > 0.35:
                    print "NB",features[i],features[j], nb_r_score, nb_p_score
                if r_score > 0.35 and p_score > 0.35:
                    print "TREE",features[i],features[j], r_score, p_score
                    #print clf.feature_importances_, features[i],features[j],features[k],features[m], r_score, p_score

print "       "
print "Exploring 2 Variable + from_this_person_to_poi_ratio Results with 10-fold CV:"
print "       "

for i in range(len(features)):
    for j in range(i+1,len(features)):
                features_list = [features[i],features[j],'from_this_person_to_poi_ratio']
                clf = DecisionTreeClassifier()
                nb = GaussianNB()
                temp = data2[features_list]
                total_predictions_tree = []
                total_predictions_nb = []
                total_labels = []
                kf = KFold(len(data2), n_folds=10,random_state=42)

                for train_index, test_index in kf:
                    X_train, X_test = temp.values[train_index], temp.values[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    clf.fit(X_train,y_train)
                    nb.fit(X_train,y_train)
                    total_predictions_tree = total_predictions_tree + clf.predict(X_test).tolist()
                    total_predictions_nb = total_predictions_nb + nb.predict(X_test).tolist()
                    total_labels = total_labels + y_test.tolist()

                r_score = recall_score(total_predictions_tree,total_labels)
                p_score = precision_score(total_predictions_tree,total_labels)
                nb_r_score = recall_score(total_predictions_nb,total_labels)
                nb_p_score = precision_score(total_predictions_nb,total_labels)
                #if nb_r_score > 0.4 and nb_p_score > 0.4:
                #    print "NB",features[i],features[j], nb_r_score, nb_p_score
                if r_score > 0.4 and p_score > 0.4:
                    print clf.feature_importances_,features[i],features[j], r_score, p_score
                    #print clf.feature_importances_, features[i],features[j],features[k],features[m], r_score, p_score


print "       "
print "Exploring 2 Variable + from_poi_to_this_person_ratio Results with 10-fold CV:"
print "       "

for i in range(len(features)):
    for j in range(i+1,len(features)):
                features_list = [features[i],features[j],'from_poi_to_this_person_ratio']
                clf = DecisionTreeClassifier()
                nb = GaussianNB()
                temp = data2[features_list]
                total_predictions_tree = []
                total_predictions_nb = []
                total_labels = []
                kf = KFold(len(data2), n_folds=10,random_state=42)

                for train_index, test_index in kf:
                    X_train, X_test = temp.values[train_index], temp.values[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    clf.fit(X_train,y_train)
                    nb.fit(X_train,y_train)
                    total_predictions_tree = total_predictions_tree + clf.predict(X_test).tolist()
                    total_predictions_nb = total_predictions_nb + nb.predict(X_test).tolist()
                    total_labels = total_labels + y_test.tolist()

                r_score = recall_score(total_predictions_tree,total_labels)
                p_score = precision_score(total_predictions_tree,total_labels)
                nb_r_score = recall_score(total_predictions_nb,total_labels)
                nb_p_score = precision_score(total_predictions_nb,total_labels)
                #if nb_r_score > 0.4 and nb_p_score > 0.4:
                #    print "NB",features[i],features[j], nb_r_score, nb_p_score
                if r_score > 0.4 and p_score > 0.4:
                    print clf.feature_importances_,features[i],features[j], r_score, p_score
                    #print clf.feature_importances_, features[i],features[j],features[k],features[m], r_score, p_score


print "       "
print "Exploring 2 Variable + from_this_person_to_poi_ratio + from_poi_to_this_person_ratio Results with 10-fold CV:"
print "       "

for i in range(len(features)):
    for j in range(i+1,len(features)):
                features_list = [features[i],features[j],'from_this_person_to_poi_ratio','from_poi_to_this_person_ratio']
                clf = DecisionTreeClassifier()
                nb = GaussianNB()
                temp = data2[features_list]
                total_predictions_tree = []
                total_predictions_nb = []
                total_labels = []
                kf = KFold(len(data2), n_folds=10,random_state=42)

                for train_index, test_index in kf:
                    X_train, X_test = temp.values[train_index], temp.values[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    clf.fit(X_train,y_train)
                    nb.fit(X_train,y_train)
                    total_predictions_tree = total_predictions_tree + clf.predict(X_test).tolist()
                    total_predictions_nb = total_predictions_nb + nb.predict(X_test).tolist()
                    total_labels = total_labels + y_test.tolist()

                r_score = recall_score(total_predictions_tree,total_labels)
                p_score = precision_score(total_predictions_tree,total_labels)
                nb_r_score = recall_score(total_predictions_nb,total_labels)
                nb_p_score = precision_score(total_predictions_nb,total_labels)
                #if nb_r_score > 0.4 and nb_p_score > 0.4:
                #    print "NB",features[i],features[j], nb_r_score, nb_p_score
                if r_score > 0.4 and p_score > 0.4:
                    print clf.feature_importances_,features[i],features[j], r_score, p_score
                    #print clf.feature_importances_, features[i],features[j],features[k],features[m], r_score, p_score

print "       "
print "Exploring 3 Variable with 10-fold CV:"
print "       "

for i in range(len(features)):
    for j in range(i+1,len(features)):
        for k in range(j+1,len(features)):
                features_list = [features[i],features[j],features[k]]
                clf = DecisionTreeClassifier()
                nb = GaussianNB()
                temp = data2[features_list]
                total_predictions_tree = []
                total_predictions_nb = []
                total_labels = []
                kf = KFold(len(data2), n_folds=10,random_state=42)

                for train_index, test_index in kf:
                    X_train, X_test = temp.values[train_index], temp.values[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    clf.fit(X_train,y_train)
                    nb.fit(X_train,y_train)
                    total_predictions_tree = total_predictions_tree + clf.predict(X_test).tolist()
                    total_predictions_nb = total_predictions_nb + nb.predict(X_test).tolist()
                    total_labels = total_labels + y_test.tolist()

                r_score = recall_score(total_predictions_tree,total_labels)
                p_score = precision_score(total_predictions_tree,total_labels)
                nb_r_score = recall_score(total_predictions_nb,total_labels)
                nb_p_score = precision_score(total_predictions_nb,total_labels)
                if r_score > 0.5 and p_score > 0.5:
                    print clf.feature_importances_,features[i],features[j],features[k], r_score, p_score

print "       "
print "Exploring 4 Variable and feature importance with 10-fold CV:"
print "       "

for i in range(len(features)):
    for j in range(i+1,len(features)):
        for k in range(j+1,len(features)):
            for m in range(k+1,len(features)):
                features_list = [features[i],features[j],features[k],features[m]]
                clf = DecisionTreeClassifier()
                nb = GaussianNB()
                temp = data2[features_list]
                total_predictions_tree = []
                total_predictions_nb = []
                total_labels = []
                kf = KFold(len(data2), n_folds=10,random_state=42)

                for train_index, test_index in kf:
                    X_train, X_test = temp.values[train_index], temp.values[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    clf.fit(X_train,y_train)
                    nb.fit(X_train,y_train)
                    total_predictions_tree = total_predictions_tree + clf.predict(X_test).tolist()
                    total_predictions_nb = total_predictions_nb + nb.predict(X_test).tolist()
                    total_labels = total_labels + y_test.tolist()

                r_score = recall_score(total_predictions_tree,total_labels)
                p_score = precision_score(total_predictions_tree,total_labels)
                nb_r_score = recall_score(total_predictions_nb,total_labels)
                nb_p_score = precision_score(total_predictions_nb,total_labels)
                if r_score > 0.5 and p_score > 0.5:
                    print clf.feature_importances_, features[i],features[j],features[k],features[m], r_score, p_score



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn import grid_search

parameters = {"criterion":("gini","entropy"),
              "min_samples_split":[1,2,4,8,16,32],
              "max_depth":[None,1,2,4,8,16,32],
              "min_samples_leaf":[1,2,4,8,16,32]
              }
best_features = data2[['expenses','other','from_this_person_to_poi']].values
dt = DecisionTreeClassifier()
clf = grid_search.GridSearchCV(dt, parameters,score_func='f1')

train_labels,test_labels,train_features,test_features = train_test_split(labels,best_features,test_size=0.50, random_state=42)
try:
    clf.fit(train_features,train_labels)
    clf = clf.best_estimator_
except:
    clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=32, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best')

print "    "
print "Running best clf in test_classifier"
print "    "


features_list = ['poi','expenses','other','from_this_person_to_poi']
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)

