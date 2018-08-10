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
dataset = pd.read_csv('.\datasets\Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

### Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

### Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Step 2 | Logistic Regression Model

### Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

## Step 3 | Predection
### Predicting the Test set results
y_pred = classifier.predict(X_test)

## Step 4 | Evaluating The Predection

### Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

## Visualization
import matplotlib.pyplot as plt

x_1 = X_train[np.where(y_train == 1)]
x_0 = X_train[np.where(y_train == 0)]
fig, ax = plt.subplots()
ax.scatter(x_0[:, 0], x_0[:, 1], color='b', label='0')
ax.scatter(x_1[:, 0], x_1[:, 1], color='r', label='1')

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()
plt.show()
