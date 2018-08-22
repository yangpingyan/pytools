#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2018/8/14 16:30 
# @Author : yangpingyan@gmail.com
# ML项目的完整流程
# 1. 项目概述。
# 2. 获取数据。
# 3. 发现并可视化数据，发现规律。
# 4. 为机器学习算法准备数据。
# 5. 选择模型，进行训练。
# 6. 微调模型。
# 7. 给出解决方案。
# 8. 部署、监控、维护系统。
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing, feature_selection, model_selection, metrics, svm
import time
import os

# to make output display better
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 1000)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# read large csv file
PROJECT_ROOT_DIR = os.getcwd()
DATA_ID = "mibaodata_ml.csv"
DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", DATA_ID)

# discount 影响很大, 0.026, 0.762
starttime = time.clock()
print("Mission start")

tmp = ['goods_name', 'goods_type', 'price', 'old_level', 'deposit',
       'cost', 'first_pay', 'daily_rent', 'lease_term', 'discount', 'pay_num',
       'freeze_money', 'disposable_payment_discount', 'original_daily_rent',
       'card_id', 'zmxy_score',  # 生成sex, age, zmf_score, xbf_score
       'phone_book', 'phone',
       'provice', 'city', 'regoin', 'type.1', 'detail_json', 'result',
       'emergency_contact_phone', 'emergency_contact_relation',
       'pay_type', 'merchant_id', 'channel', 'type', 'source',  # 影响比较小的元素
       'ip', 'order_type', 'receive_address',  # 可能有用但还不知道怎么用
       'releted', ]  # 丢弃相关数据

df = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
df.fillna(value=0, inplace=True)

features_cat = ['sex', 'pay_num', 'disposable_payment_discount', 'phone_book']
features_number = ['cost', 'age', 'discount', 'deposit', 'freeze_money','zmf_score', 'xbf_score', ]
df_num = df[features_number]
df_discrete = df[features_cat]
y = df['check_result']

# 数据调试代码
# df.sort_values(by=['phone_book'], inplace=True)
# df[df['phone_book'].isnull()]
# df['deposit'].value_counts()
# df.info()



# 特征处理
for feature in features_cat:
    df[feature] = preprocessing.LabelEncoder().fit_transform(df[feature])

## Importing the dataset
x = df[features_cat + features_number]


## Handling the missing data
# x = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0).fit_transform(x)

## Feature Scaling
x[features_number] = preprocessing.StandardScaler().fit_transform(x[features_number])

## Encoding Categorical data
# x = preprocessing.OneHotEncoder(categorical_features=np.array([0,1,2,3,4,5])).fit_transform(x).toarray()


## Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20, random_state=0)

## Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

## Fitting SVM to the Training set
classifier = svm.SVC(kernel='rbf', random_state=0)
# feature_selection.RFE(estimator=classifier, n_features_to_select=2).fit_transform(x_train, y_train)
classifier.fit(x_train, y_train)
## Predicting the Test set results
y_pred = classifier.predict(x_test)
print("squared mean squared error:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
## Making the Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
print("percision score:{:.3f}".format(metrics.precision_score(y_test, y_pred))) #0.930
print("recall score:{:.3f}".format(metrics.recall_score(y_test, y_pred)))    #0.706
# print(cm / np.sum(cm, axis=1))

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


print("ML mission complete! {:.2f}S".format((time.clock() - starttime)))
