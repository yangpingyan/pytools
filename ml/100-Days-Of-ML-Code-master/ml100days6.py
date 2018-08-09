#!/usr/bin/env  
# -*- coding: utf-8 -*- 
# @Time : 2018/8/9 17:46 
# @Author : yangpingyan@gmail.com

# Logistic Regression

## Step 1 | Data Pre-Processing

### Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing the dataset


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


### Splitting the dataset into the Training set and Test set


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


### Feature Scaling


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Step 2 | Logistic Regression Model

The library for this job which is going to be the linear model library and it is called linear because the logistic regression is a linear classifier which means that here since we're in two dimensions, our two categories of users are going to be separated by a straight line. Then import the logistic regression class.
Next we will create a new object from this class which is going to be our classifier that we are going to fit on our training set.

### Fitting Logistic Regression to the Training set


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

## Step 3 | Predection

### Predicting the Test set results


y_pred = classifier.predict(X_test)


## Step 4 | Evaluating The Predection

We predicted the test results and now we will evaluate if our logistic regression model learned and understood correctly.
So this confusion matrix is going to contain the correct predictions that our model made on the set as well as the incorrect predictions.

### Making the Confusion Matrix


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


## Visualization

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Other%20Docs/training.png">
</p> 

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Other%20Docs/testing.png">
</p> 
