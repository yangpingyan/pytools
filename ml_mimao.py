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
from sklearn import preprocessing, feature_selection, model_selection, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
features_number = ['cost', 'age', 'deposit', 'freeze_money', 'zmf_score', 'xbf_score', ]
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
svm_clf = SVC()
svm_clf.fit(x_train, y_train)
y_train_pred = svm_clf.predict(x_train)
cm = metrics.confusion_matrix(y_train, y_train_pred)
print(cm)
print("squared mean squared error:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("precision score:{:.3f}".format(metrics.precision_score(y_train, y_train_pred)))  # 0.930
print("recall score:{:.3f}".format(metrics.recall_score(y_train, y_train_pred)))  # 0.706
print("SVC ML mission complete! {:.2f}S".format((time.clock() - starttime)))

print("SVC:")
starttime = time.clock()
svm_clf = SVC()
y_train_pred = model_selection.cross_val_predict(svm_clf, x_train, y_train, cv=10)
cm = metrics.confusion_matrix(y_train, y_train_pred)
print(cm)
print("squared mean squared error:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("precision score:{:.3f}".format(metrics.precision_score(y_train, y_train_pred)))  # 0.930
print("recall score:{:.3f}".format(metrics.recall_score(y_train, y_train_pred)))  # 0.706
print("SVC ML mission complete! {:.2f}S".format((time.clock() - starttime)))

print("RandomForestClassifier:")
starttime = time.clock()
forest_clf = RandomForestClassifier()
y_train_pred = model_selection.cross_val_predict(forest_clf, x_train, y_train, cv=10)
cm = metrics.confusion_matrix(y_train, y_train_pred)
print(cm)
print("squared mean squared error:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("precision score:{:.3f}".format(metrics.precision_score(y_train, y_train_pred)))  # 0.930
print("recall score:{:.3f}".format(metrics.recall_score(y_train, y_train_pred)))  # 0.706
print("RandomForestClassifier ML mission complete! {:.2f}S".format((time.clock() - starttime)))
forest_clf.fit(x_train, y_train)
for name, score in zip(x.columns.tolist(), forest_clf.feature_importances_):
    print(name, score)

print("DecisionTreeClassifier:")
starttime = time.clock()
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
y_train_pred = model_selection.cross_val_predict(tree_clf, x_train, y_train, cv=10)
cm = metrics.confusion_matrix(y_train, y_train_pred)
print(cm)
print("squared mean squared error:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("precision score:{:.3f}".format(metrics.precision_score(y_train, y_train_pred)))  # 0.930
print("recall score:{:.3f}".format(metrics.recall_score(y_train, y_train_pred)))  # 0.706
print("DecisionTreeClassifier ML mission complete! {:.2f}S".format((time.clock() - starttime)))


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

for clf in (log_clf, rnd_clf, svm_clf):
    starttime = time.clock()
    clf.fit(x_train, y_train)
    endtime = time.clock()
    y_pred = clf.predict(x_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)
    print("{} run time {:.3f}".format(clf.__class__.__name__,endtime-starttime))
    print("accuracy score:{:.3f}".format(metrics.accuracy_score(y_test, y_pred)))
    print("precision score:{:.3f}".format(metrics.precision_score(y_test, y_pred)))
    print("recall score:{:.3f}".format(metrics.recall_score(y_test, y_pred)))
    print("squared mean squared error:{:.3f}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

exit(0)
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
forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# The best hyperparameter combination found:
grid_search.best_params_
grid_search.best_estimator_

# Let's look at the score of each hyperparameter combination tested during the grid search:
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)

# 模型微调-随机搜索
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)



# 分析最佳模型和它们的误差
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# 用测试集评估系统
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse

# 模型保存于加载
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")




# 然后就是项目的预上线阶段：你需要展示你的方案（重点说明学到了什么、做了什么、没做
# 什么、做过什么假设、系统的限制是什么，等等），记录下所有事情，用漂亮的图表和容易
# 记住的表达（比如，“收入中位数是房价最重要的预测量”）做一次精彩的展示。