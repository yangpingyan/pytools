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
from sklearn import metrics

# discount 影响很大


tmp = ['added_service', 'first_pay', 'channel',
       'pay_type', 'merchant_id', 'lease_term', 'daily_rent', 'accident_insurance', 'type',
       'ip', 'releted', 'order_type', 'delivery_way', 'source',
       'lease_num', 'original_daily_rent',
       'provice', 'city', 'regoin', 'receive_address', 'emergency_contact_name', 'phone_book',
       'emergency_contact_phone', 'emergency_contact_relation', 'type.1', 'detail_json',
       'goods_type']  # 最后一排特征对预测结果影响不明显

csv.field_size_limit(100000000)
df_ml = pd.read_csv("mibaodata_ml.csv", encoding='utf-8', engine='python')
df = df_ml.drop_duplicates(subset=['card_id'], keep='last')

# 数据调试代码
# df.sort_values(by=['phone_book'], inplace=True)
# df[df['phone_book'].isnull()]
# df['phone_book'].value_counts()


features_label = ['zmf_score', 'xbf_score', 'sex', 'pay_num', 'disposable_payment_discount', 'phone_book']
features_number = ['cost', 'age', 'discount', 'deposit', 'freeze_money', ]
features_label = ['zmf_score', 'xbf_score', 'disposable_payment_discount']
features_number = [ 'discount' ]

df = df[['check_result'] + features_label + features_number]
print("Alldata for ML: {}".format(df.shape))

# 过滤数据
# df = df[df['zmf_score'] > 0][df['xbf_score'] > 0]
# df = df[df['cost'] > 0]
# df.dropna(subset=['pay_num'], inplace=True)
print("After handling data: {}".format(df.shape))

# 特征处理
# df['discount'].fillna(value=0, inplace=True)
# df['disposable_payment_discount'].fillna(value=0, inplace=True)
# df['deposit'].fillna(value=0, inplace=True)
# df['freeze_money'].fillna(value=0, inplace=True)
# df['phone_book'] = df['phone_book'].map(lambda x: 1 if isinstance(x, str) else 0)
df.fillna(value=0, inplace=True)
df['disposable_payment_discount'] = preprocessing.LabelEncoder().fit_transform(df['disposable_payment_discount'])

## Importing the dataset
x = df[features_label + features_number]
y = df['check_result']

## Handling the missing data
# x = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0).fit_transform(x)

## Encoding Categorical data

# x = preprocessing.OneHotEncoder(categorical_features='all').fit_transform(x).toarray()

## Feature Scaling
x[features_number] = preprocessing.StandardScaler().fit_transform(x[features_number])

## Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

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
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(cm / np.sum(cm, axis=1))

# y_train_pred = classifier.predict(x_train)
# cm_train = confusion_matrix(y_train, y_train_pred)

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

print("ML mission complete!")
