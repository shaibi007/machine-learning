#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:59:58 2017

@author: shoaibrafique
"""

#Simple Linear Regression

import numpy as np   #numpy is the library and np is short name
import matplotlib.pyplot as plt   #plotting charts
import pandas as pd                 #import and manage datasets

print ("Success importing the libraries")

#Import the datasets
dataset = pd.read_csv('/Users/shoaibrafique/Downloads/machine learning/Multiple Linear Regression/icecream.csv', encoding='utf-8')
print ("Success importing the dataset")

#Droping the non-required columns (price and index)
dataset = dataset.drop(['price','Unnamed: 0'],axis=1)
                  
#Create matrix of features - columns of independant variables
X = dataset.iloc[:,1:].values
               
#create dependant variable vector
y = dataset.iloc[:,0].values    

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
                 
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward elimination -- removing not significant statistically independent variable
import statsmodels.formula.api as sm

#Checking the p values of the independant vriables
X_opt = X[:, [ 0,1]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()   #OLS('dependent variables', 'matrix of features')
#look for the highest p value -- use a library function to get summary .. remove the highest p values (which are greater than the significant value of 5%)
regressor_ols.summary()

"""when we look at the summary we see that p values are 0 which means that all the independant variables are significant for the model
If that weren't the case, we would have proceeded as follows
y=b0+b1x1+-----bnxn  --> the library doesn't take care of x0 
y=b0x0+b1x1+-----bnxn .. so add a columns of 1 to X 
X = np.append(arr=np.ones((30,1)).astype(int), values=X, axis = 1)
and then proceed with backward elimination to get rid of all the variables with p values greater than the significance value

#Backward Elimination
X_opt = X[:, [1,2]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()   #OLS('dependent variables', 'matrix of features')
#look for the highest p value -- use a library function to get summary
regressor_ols.summary()"""

#Hence the model is optimal and it can be seen that the predicted values (y_pred) are almost equal to the test set (y_test) 

#The following graph shows the correlation between the test set and the predicted values

plt.plot(y_test,'bo')
plt.plot(y_pred,'ro')
plt.title('The correlation between the test set and the predicted values')
plt.legend(['test set', 'predicted values'])
plt.show()