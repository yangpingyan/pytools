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
import time
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import lightgbm as lgb

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
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
# Get Data
df = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
print("ML初始数据量: {}".format(df.shape))

x = df.drop(['check_result'], axis=1)
y = df['check_result']
## Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

'''
With these two criteria - Supervised Learning plus Classification and Regression, 
we can narrow down our choice of models to a few. These include:
Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine
XGBoost
LightGBM
'''


# 保存所有模型得分
def add_score(score_df, name, runtime, x_test, y_test):
    score_df[name] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred),
                      f1_score(y_test, y_pred), runtime, confusion_matrix(y_test, y_pred)]

    return score_df


log_clf = LogisticRegression()
log_clf.fit(x_train, y_train)
y_pred = log_clf.predict(x_test)
coeff_df = pd.DataFrame(x_test.columns.values)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(log_clf.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False, inplace=True)
print(coeff_df)

knn_clf = KNeighborsClassifier()
gaussian_clf = GaussianNB()
perceptron_clf = Perceptron()
sgd_clf = SGDClassifier(max_iter=5)
svm_clf = SVC(probability=True)
linear_svc = LinearSVC()
decision_tree = DecisionTreeClassifier()
rnd_clf = RandomForestClassifier()
xg_clf = XGBClassifier()
lgbm = lgb.LGBMClassifier()

clf_list = [lgbm, xg_clf, knn_clf, log_clf, sgd_clf, decision_tree, rnd_clf, gaussian_clf, linear_svc]
score_df = pd.DataFrame(index=['accuracy', 'precision', 'recall', 'f1', 'runtime', 'confusion_matrix'])
for clf in clf_list:
    starttime = time.clock()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    add_score(score_df, clf.__class__.__name__, time.clock() - starttime, x_test, y_test)

print(score_df)

# 使用PR曲线： 当正例较少或者关注假正例多假反例。 其他情况用ROC曲线
plt.figure(figsize=(8, 6))
plt.xlabel("Recall(FPR)", fontsize=16)
plt.ylabel("Precision(TPR)", fontsize=16)
plt.axis([0, 1, 0, 1])
color = ['r', 'y', 'b', 'g', 'c']
for cn, clf in enumerate((knn_clf, rnd_clf, xg_clf)):
    print(clf.__class__.__name__)
    y_train_pred = cross_val_predict(clf, x_train, y_train, cv=3)
    if clf in (rnd_clf, knn_clf, decision_tree, gaussian_clf, xg_clf):
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
grid_search = GridSearchCV(forest_clf, param_grid, cv=5, scoring='roc_auc', n_jobs=1, return_train_score=True)
starttime = time.clock()
grid_search.fit(x_train, y_train)
print(time.clock() - starttime)
# The best hyperparameter combination found:
grid_search.best_params_
grid_search.best_estimator_
grid_search.best_score_

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
rnd_search.fit(x_train, y_train)
print(time.clock() - starttime)
rnd_search.best_params_
rnd_search.best_estimator_
rnd_search.best_score_
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
importance_df = pd.DataFrame({'name': x_train.columns, 'importance': feature_importances})
importance_df.sort_values(by=['importance'], ascending=False, inplace=True)
print(importance_df)

# 分析最佳模型和它们的误差

# 用测试集评估系统
clf = grid_search.best_estimator_
starttime = time.clock()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
add_score(score_df, clf.__class__.__name__ + 'best', time.clock() - starttime, x_test, y_test)
print(score_df)
# 模型保存于加载
# from sklearn.externals import joblib
#
# joblib.dump(my_model, "my_model.pkl")
# my_model_loaded = joblib.load("my_model.pkl")

# 然后就是项目的预上线阶段：你需要展示你的方案（重点说明学到了什么、做了什么、没做
# 什么、做过什么假设、系统的限制是什么，等等），记录下所有事情，用漂亮的图表和容易
# 记住的表达（比如，“收入中位数是房价最重要的预测量”）做一次精彩的展示。
