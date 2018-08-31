#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/8/31 10:46 
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
from sklearn.preprocessing import StandardScaler
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

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30)
X_train, X_test, y_train, y_test = train_test_split(X, y)
# plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=0.4)
# plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=0.4)

knn_clf = KNeighborsClassifier()
log_clf = LogisticRegression()
sgd_clf = SGDClassifier(max_iter=5)
svm_clf = SVC(probability=True)
rnd_clf = RandomForestClassifier()
voting_hard_clf = VotingClassifier(
    estimators=[('knn', log_clf),  ('lr', log_clf), ('sf', sgd_clf), ('svc', svm_clf), ('rf', rnd_clf)],
    voting='hard')
voting_soft_clf = VotingClassifier(
    estimators=[('knn', log_clf), ('lr', log_clf), ('svc', svm_clf), ('rf', rnd_clf)],
    voting='soft')  # 采用分类的probability

for clf in (knn_clf, log_clf, sgd_clf, svm_clf, rnd_clf, voting_hard_clf, voting_soft_clf):
    starttime = time.clock()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
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
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
    if clf is rnd_clf or clf is knn_clf:
        y_probas = cross_val_predict(clf, X_train, y_train, cv=3, method="predict_proba")
        y_scores = y_probas[:, 1]  # score = proba of positive class
    else:
        y_scores = cross_val_predict(clf, X_train, y_train, cv=3, method="decision_function")

    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plt.plot(recalls, precisions, linewidth=1, label=clf.__class__.__name__, color=color[cn])
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    print("{} roc socore: {}".format(clf.__class__.__name__, roc_auc_score(y_train, y_scores)))
    plt.plot(fpr, tpr, linewidth=1, color=color[cn])

plt.legend()
plt.show()

# # 误差分析
cm = confusion_matrix(y_train, y_train_pred)

def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


plt.matshow(cm, cmap=plt.cm.gray)
plt.show()

row_sums = cm.sum(axis=1, keepdims=True)
norm_conf_mx = cm / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()





# 可以通过决策分数来间接设置阈值来改变准确率和召回率
# decision_function 得到分数
y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function")

# hack to work around issue #9589 in Scikit-Learn 0.19.0
if y_scores.ndim == 2:
    y_scores = y_scores[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
plt.show()

(y_train_pred == (y_scores > 0)).all()
y_train_pred_90 = (y_scores > 70000)
precision_score(y_train, y_train_pred_90)
recall_score(y_train, y_train_pred_90)
