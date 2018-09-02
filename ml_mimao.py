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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import time
import os
import csv

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

tmp = [
       'first_pay', 'daily_rent',
       'provice', 'city', 'regoin', 'type.1', 'detail_json', 'result',
       'ip', 'receive_address',  # 可能有用但还不知道怎么用
        ]  # 丢弃相关数据

df = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
df.fillna(value=0, inplace=True)
print("ML初始数据量: {}".format(df.shape))
# 去除身份证重复的订单：
df.drop_duplicates(subset=['card_id'], keep='last', inplace=True)
print("去除身份证重复的订单后的数据量: {}".format(df.shape))
features_cat = ['sex', 'pay_num', 'disposable_payment_discount', 'phone_book', 'create_time_cat', 'old_level',
                'emergency_contact_phone', 'source', 'order_type',  ]
features_number = ['cost', 'price', 'age', 'deposit', 'freeze_money', 'zmf_score', 'xbf_score', 'original_daily_rent', 'first_pay']
df_num = df[features_number]
df_cat = df[features_cat]

# 数据调试代码
# df.sort_values(by=['phone_book'], inplace=True)
# df[df['phone_book'].isnull()]
# df['deposit'].value_counts()
# df.info()

# 特征处理
for feature in features_cat:
    df[feature] = LabelEncoder().fit_transform(df[feature])

## Feature Scaling
df[features_number] = StandardScaler().fit_transform(df[features_number])

x = df[features_cat + features_number]
y = df['check_result']
## Encoding Categorical data
# x = preprocessing.OneHotEncoder(categorical_features=np.array([0,1,2,3,4,5])).fit_transform(x).toarray()


## Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

knn_clf = KNeighborsClassifier()
log_clf = LogisticRegression()
sgd_clf = SGDClassifier(max_iter=5)
svm_clf = SVC(probability=True)
rnd_clf = RandomForestClassifier()
voting_hard_clf = VotingClassifier(
    estimators=[('knn', log_clf), ('lr', log_clf), ('sf', sgd_clf), ('svc', svm_clf), ('rf', rnd_clf)],
    voting='hard')
voting_soft_clf = VotingClassifier(
    estimators=[('knn', log_clf), ('lr', log_clf), ('svc', svm_clf), ('rf', rnd_clf)],
    voting='soft')  # 采用分类的probability

for clf in (knn_clf, log_clf, sgd_clf, svm_clf, rnd_clf, voting_hard_clf, voting_soft_clf):
    starttime = time.clock()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, time.clock() - starttime)
    print(confusion_matrix(y_test, y_pred))
    print("accuracy_score:{:.3f}".format(accuracy_score(y_test, y_pred)))
    print("precision_score:{:.3f}".format(precision_score(y_test, y_pred)))
    print("recall_score:{:.3f}".format(recall_score(y_test, y_pred)))
    print("f1_score:{:.3f}".format(f1_score(y_test, y_pred)))

# 使用PR曲线： 当正例较少或者关注假正例多假反例。 其他情况用ROC曲线
plt.figure(figsize=(8, 6))
plt.xlabel("Recall(FPR)", fontsize=16)
plt.ylabel("Precision(TPR)", fontsize=16)
plt.axis([0, 1, 0, 1])
color = ['r', 'y', 'b', 'g', 'c']
for cn, clf in enumerate((knn_clf, log_clf, sgd_clf, svm_clf, rnd_clf)):
    y_train_pred = cross_val_predict(clf, x_train, y_train, cv=3)
    if clf is rnd_clf or clf is knn_clf:
        y_probas = cross_val_predict(clf, x_train, y_train, cv=3, method="predict_proba", n_jobs=-1)
        y_scores = y_probas[:, 1]  # score = proba of positive class
    else:
        y_scores = cross_val_predict(clf, x_train, y_train, cv=3, method="decision_function", n_jobs=-1)

    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plt.plot(recalls, precisions, linewidth=1, label=clf.__class__.__name__, color=color[cn])
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    print("{} roc socore: {}".format(clf.__class__.__name__, roc_auc_score(y_train, y_scores)))
    plt.plot(fpr, tpr, linewidth=1, color=color[cn])

plt.legend()
plt.show()

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

# 模型微调，寻找最佳超参数
# 网格搜索
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_clf = RandomForestClassifier()
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, return_train_score=True)
starttime = time.clock()
grid_search.fit(x, y)
print(time.clock() - starttime)
# The best hyperparameter combination found:
grid_search.best_params_
grid_search.best_estimator_
grid_search.best_score_
grid_search.grid_scores_

# Let's look at the score of each hyperparameter combination tested during the grid search:
cvres = grid_search.cv_results_
results = pd.DataFrame(grid_search.cv_results_)
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# 模型微调-随机搜索
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_clf = RandomForestClassifier()
rnd_search = RandomizedSearchCV(forest_clf, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='roc_auc', n_jobs=-1)
starttime = time.clock()
rnd_search.fit(x, y)
print(time.clock() - starttime)
rnd_search.best_params_
rnd_search.best_estimator_
rnd_search.best_score_
rnd_search.grid_scores_
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# 分析最佳模型和它们的误差
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
for name, score in zip(x.columns, grid_search.best_estimator_.feature_importances_):
    print(name, score)
# 用测试集评估系统


# 模型保存于加载
# from sklearn.externals import joblib
#
# joblib.dump(my_model, "my_model.pkl")
# my_model_loaded = joblib.load("my_model.pkl")

# 然后就是项目的预上线阶段：你需要展示你的方案（重点说明学到了什么、做了什么、没做
# 什么、做过什么假设、系统的限制是什么，等等），记录下所有事情，用漂亮的图表和容易
# 记住的表达（比如，“收入中位数是房价最重要的预测量”）做一次精彩的展示。
