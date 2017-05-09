#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:41:28 2017

@author: shoaibrafique
"""
# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/shoaibrafique/Desktop/Machine Learning/Udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
