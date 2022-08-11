# importing libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


data = pd.read_csv("Obfuscated-MalMem2022.csv")  # importing the csv file from folder
X = data.iloc[:, 1:56]   # select original features data
y = data.loc[:, 'Class']  # select target label
le = preprocessing.LabelEncoder()  # Label Encoder function call
le.fit(y)  # fitting to target label
y = le.transform(y)  # translate the label from string to int

# split the dataset as training set and testing set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# multi-perceptron neural network classifier
clf = MLPClassifier(hidden_layer_sizes=(15, 10), random_state=1, validation_fraction=0.2, max_iter=100)  # multi-perceptron neural network classifier function call
clf = clf.fit(X_train, y_train)  # training using multi-perceptron neural network classifier
# predict the test data
y_pred = clf.predict(X_test)  # predict the label of test dataset
print("------------------------------------------------------------")
print("multi-perceptron neural network classifier")
print(classification_report(y_test, y_pred, digits=4))  # display the classification report.
print(confusion_matrix(y_test, y_pred, labels=range(2)))  # display the confusion matrix
print(cohen_kappa_score(y_test, y_pred))  # print the kappa statistics of MLP classifier

# Random Forest classifier
clf2 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)  # training using Random Forest classifier
y_pred2 = clf2.predict(X_test)  # predict the label of test dataset
print("------------------------------------------------------------")
print("Random Forest classifier")
print(classification_report(y_test, y_pred2, digits=4))  # display the classification report.
print(confusion_matrix(y_test, y_pred2, labels=range(2)))  # display the confusion matrix
print(cohen_kappa_score(y_test, y_pred2))  # print the kappa statistics of Random Forest classifier

# J48 classifier
clf3 = tree.DecisionTreeClassifier()  # J48 classifier function call
clf3 = clf3.fit(X_train, y_train)  # training using J48 classifier
y_pred3 = clf3.predict(X_test)  # predict the label of test dataset
print("------------------------------------------------------------")
print("J48 classifier")
print(classification_report(y_test, y_pred3, digits=4))  # display the classification report.
print(confusion_matrix(y_test, y_pred3, labels=range(2)))  # display the confusion matrix
print(cohen_kappa_score(y_test, y_pred3))  # print the kappa statistics of J48 classifier

# Logistic Regression classifier
print("------------------------------------------------------------")
print("Logistic Regression classifier")
clf4 = LogisticRegression(random_state=0, max_iter=200).fit(X_train, y_train)  # training using Logistic Regression classifier
y_pred4 = clf4.predict(X_test)  # predict the label of test dataset
print(classification_report(y_test, y_pred4, digits=4))  # display the classification report.
print(confusion_matrix(y_test, y_pred4, labels=range(2)))  # display the confusion matrix
print(cohen_kappa_score(y_test, y_pred4))  # print the kappa statistics of Logistic Regression classifier
