#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
            'poi',
            "deferral_payments",
            "deferred_income",
            "director_fees",
            "exercised_stock_options",
            "expenses",
            "from_messages",
            "from_poi_to_this_person",
            "from_this_person_to_poi",
            "long_term_incentive",
            "loan_advances",
            "other",
            "bonus",
            "restricted_stock",
            "restricted_stock_deferred",
            "salary",
            "shared_receipt_with_poi",
            "to_messages",
            "total_payments",
            "total_stock_value"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print('Number of data points in the set:', len(data_dict))

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

print("Number of data points in the dataset after removing 'TOTAl and THE TRAVEL AGENCY IN THE PARK': ", len(data_dict))

  
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for person in my_dataset.values():
    bonus_to_salary_ratio = 0
    if person['salary'] != 'NaN' and person['bonus'] != 'NaN':
        person["bonus_to_salary_ratio"] = float(person['bonus'])/float(person['salary'])
    else: 
        person['bonus_to_salary_ratio'] = 0

    person['to_poi_message_ratio'] = 0
    person['from_poi_message_ratio'] = 0
    if float(person['from_messages']) > 0:
        person['to_poi_message_ratio'] = float(person['from_this_person_to_poi'])/float(person['from_messages'])
    if float(person['to_messages']) > 0:
        person['from_poi_message_ratio'] = float(person['from_poi_to_this_person'])/float(person['to_messages'])

features_list.extend(['bonus_to_salary_ratio', 'to_poi_message_ratio', 'from_poi_message_ratio'])
my_dataset
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#scaling and feature_selection
scaler = MinMaxScaler()
skb = SelectKBest(k = 'all')

#naive bayes
#clf = GaussianNB()
#clf =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", GaussianNB())])

#SVM
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf')
clf = Pipeline(steps=[('scaling',scaler),("SKB", skb), ("SVM", SVC())])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print('accuracy:', accuracy)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)