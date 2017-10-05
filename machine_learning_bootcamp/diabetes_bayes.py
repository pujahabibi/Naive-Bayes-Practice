import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

diabetes_dataset = pd.read_csv("diabetes.csv")
feature_dataset = diabetes_dataset[['times_pregnant','glucose_concentration','diastolic_blood_pressure',
                                    'triceps_skin_fold','serum_insulin','bmi','diabetes_pedigree','age']]
target_dataset = diabetes_dataset['class']

feature_train, feature_test, target_train, target_test = train_test_split(feature_dataset, target_dataset, test_size=0.4,
                                                                          random_state=101)
gaussian_bayes_classifier = GaussianNB()
logistic_regression_classifier = LogisticRegression()
multilayer_classifier = MLPClassifier()
bagging_classifier = DecisionTreeClassifier()

print(gaussian_bayes_classifier.fit(feature_train, target_train))
print(logistic_regression_classifier.fit(feature_train, target_train))
print(multilayer_classifier.fit(feature_train, target_train))
print(bagging_classifier.fit(feature_train, target_train))

print(gaussian_bayes_classifier.predict(feature_test))
print()
print(logistic_regression_classifier.predict(feature_test))
print()
print(multilayer_classifier.predict(feature_test))
print()
print(bagging_classifier.predict(feature_test))
print()

print(classification_report(target_test, bagging_classifier.predict(feature_test)))
print(classification_report(target_test, multilayer_classifier.predict(feature_test)))
print(classification_report(target_test, logistic_regression_classifier.predict(feature_test)))
print(classification_report(target_test, gaussian_bayes_classifier.predict(feature_test)))
print()
print(accuracy_score(target_test, bagging_classifier.predict(feature_test)))
print(accuracy_score(target_test, multilayer_classifier.predict(feature_test)))
print(accuracy_score(target_test, logistic_regression_classifier.predict(feature_test)))
print(accuracy_score(target_test, gaussian_bayes_classifier.predict(feature_test)))
print()
print(confusion_matrix(target_test, gaussian_bayes_classifier.predict(feature_test)))

