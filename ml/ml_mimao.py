#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2018/8/14 16:30 
# @Author : yangpingyan@gmail.com
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from matplotlib.colors import ListedColormap

tmp = ['discount', 'installment', 'added_service', 'first_pay', 'full', 'channel',
       'pay_type', 'merchant_id', 'lease_term', 'daily_rent', 'accident_insurance', 'type',
       'freeze_money', 'ip', 'releted', 'order_type', 'delivery_way', 'source', 'disposable_payment_discount',
       'disposable_payment_enabled', 'lease_num', 'original_daily_rent', 'deposit', 'zmxy_score', 'card_id',
       'contact', 'phone', 'provice', 'city', 'regoin', 'receive_address', 'emergency_contact_name', 'phone_book',
       'emergency_contact_phone', 'emergency_contact_relation', 'type.1', 'detail_json', 'cost', 'discount', 'pay_num',
       'goods_type']
csv.field_size_limit(100000000)
df_ml = pd.read_csv("mibaodata_ml.csv", encoding='utf-8', engine='python')

features = ['zmf_score', 'xbf_score', 'age', 'sex']
df = df_ml[['check_result'] + features]

# 过滤数据
# df = df[df['zmf_score']>0][df['xbf_score']>0]

# 特征处理

## Importing the dataset
x = df[features]
y = df['check_result']

## Handling the missing data
# imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
# imputer = imputer.fit(X)
# X = imputer.transform(X)

## Encoding Categorical data
# labelencoder = preprocessing.LabelEncoder()
# x = labelencoder.fit_transform(x)
onehotencoder = preprocessing.OneHotEncoder(categorical_features='all')
x = onehotencoder.fit_transform(x).toarray()

## Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

## Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

## Fitting SVM to the Training set
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)
## Predicting the Test set results
y_pred = classifier.predict(x_test)

## Making the Confusion Matrix
cm_test = confusion_matrix(y_test, y_pred)
y_train_pred = classifier.predict(x_train)
cm_train = confusion_matrix(y_train, y_train_pred)

## Visualising the Training set results
## Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.5, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('SVM (Test set)')
# plt.xlabel('zmf')
# plt.ylabel('xbf')
# plt.legend()
# plt.show()
