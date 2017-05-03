#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:00:25 2017

@author: shoaibrafique
"""

#Simple Linear Regression

import numpy as np   #numpy is the library and np is short name
import matplotlib.pyplot as plt   #plotting charts
import pandas as pd                 #import and manage datasets

print ("Success importing the libraries")

#Import the datasets
dataset = pd.read_csv("/Users/shoaibrafique/Downloads/machine-learning/Linear regression/airmiles.csv").drop(['Unnamed: 0'],axis=1)
print ("Success importing the dataset")

#Create matrix of features - columns of independant variables
X = dataset.iloc[:,:-1].values
               
#create dependant variable vector
Y = dataset.iloc[:,1].values
                
#Splitting the dataset into trainig and test set
#Import cross validation librrary
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.33, random_state = 0)

#Fitting the Simple Linear Regression to Training set using a library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


"""Predicting the Test set Results
Vector - that will contain the predicted values"""
Y_pred = regressor.predict(X_test)

#Visualising the Training set Results
plt.scatter(X_train, Y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Year vs Number of Passengers (Training set)')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.show()

#Visualising the Test set Results
plt.scatter(X_test, Y_test, color ='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Year vs Number of Passengers (Test set)')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.show()

