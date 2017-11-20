#!/usr/bin/python

import sys
import cPickle as pickle
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'loan_advances',
                'bonus', 'deferred_income', 'expenses','long_term_incentive',
                'other', 'total_payments', 'director_fees',
                'exercised_stock_options', 'total_stock_value',
                'restricted_stock', 'restricted_stock_deferred',
                'to_messages', 'combined_pay_and_stock',
                'from_poi_to_this_person','from_messages',
                'from_this_person_to_poi','shared_receipt_with_poi',
                'proportion_pay_to_salary']
# kept most of the features because I use univariate feature selection later,
# removed 'email_address' as throwing an error (not able to convert to float)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL") # removes the "total" outlier that sums up financials


### Task 3: Create new feature(s)
my_dataset = data_dict
# New features:
# 1: Combine totals for payments and stock values.
# 2: Proportion of payments made above base salary to base salary
for key, value in my_dataset.iteritems():
    total_payments = value['total_payments']
    total_stock_value = value['total_stock_value']
    salary = value['salary']

    #create combined_payments_and_stock_value feature
    if total_payments != 'NaN' and total_stock_value != 'NaN':
        value['combined_pay_and_stock'] = total_payments + total_stock_value
    # If there is no total_payment value, set combined to stock value,
    # which will then be NaN if stock is also NaN
    elif total_payments == 'NaN':
        value['combined_pay_and_stock'] = total_stock_value
    # If there is a total_payment value but stock is NaN,
    # set combined value to payment value
    else:
        value['combined_pay_and_stock'] = total_payments

    # create proportion_payments_to_salary feature:
    if total_payments != 'NaN' and salary != 'NaN':
        value['proportion_pay_to_salary'] = (total_payments - salary) / salary
    # If either value is NaN, set proportion_payments_to_salary to NaN
    else:
        value['proportion_pay_to_salary'] = 'NaN'

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.


estimators = [('scaler', MinMaxScaler()), ('skb', SelectKBest()),
                ('gnb', GaussianNB())]
pipe = Pipeline(estimators)

param_grid = dict(skb__k=range(2,7))


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using tester.py

cv = StratifiedShuffleSplit(labels, test_size=0.3, random_state = 42)

gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='f1_weighted')

gs.fit(features, labels)

clf = gs.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
