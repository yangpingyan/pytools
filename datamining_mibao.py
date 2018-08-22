#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2018/8/20 17:48
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
import time
import os
from sklearn import preprocessing
from timeit import timeit

# to make output display better
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 1000)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# read large csv file
csv.field_size_limit(100000000)

# Datasets info
PROJECT_ROOT_DIR = os.getcwd()
DATA_ID = "学校数据.csv"
DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", DATA_ID)
df_alldata = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
print("原始数据量: {}".format(df_alldata.shape))

df = df_alldata.dropna(axis=1, how='all')
# 处理身份证号
df['card_id'] = df['card_id'].map(lambda x: x.replace(x[10:16], '******') if isinstance(x, str) else x)

# 取可能有用的数据
features = ['goods_name', 'goods_type', 'price', 'old_level', 'deposit',
            'cost', 'first_pay', 'daily_rent', 'lease_term', 'discount', 'pay_num',
            'freeze_money', 'disposable_payment_discount', 'original_daily_rent',
            'card_id', 'zmxy_score',  # 生成sex, age, zmf_score, xbf_score
            'phone_book', 'phone',
            'provice', 'city', 'regoin', 'type.1', 'detail_json', 'result',
            'emergency_contact_phone', 'emergency_contact_relation',
            'pay_type', 'merchant_id', 'channel', 'type', 'source',  # 影响比较小的元素
            'ip', 'order_type', 'receive_address',  # 可能有用但还不知道怎么用
            'releted', ]  # 丢弃相关数据

result = ['state', 'cancel_reason', 'check_result', 'check_remark']
df = df[result + features]
print("筛选出所有可能有用特征后的数据量: {}".format(df.shape))

# 丢弃身份证号为空的数据
df.dropna(subset=['card_id'], inplace=True)
print("去除无身份证号后的数据量: {}".format(df.shape))

# 取有审核结果的数据
df = df[df['check_result'].str.contains('SUCCESS|FAILURE', na=False)]
print("去除未经机审用户后的数据量: {}".format(df.shape))

# 去除只有唯一值的列
cols = []
for col in df.columns:
    if len(df[col].unique()) <= 1:
        cols.append(col)
df.drop(cols, axis=1, errors='ignore', inplace=True)
print("去除特征值中只有唯一值后的数据量: {}".format(df.shape))

# 去除测试数据和内部员工数据
df = df[df['cancel_reason'].str.contains('测试|内部员工') != True]
df = df[df['check_remark'].str.contains('测试|内部员工') != True]
print("去除测试数据和内部员工后的数据量: {}".format(df.shape))

# 去掉用户自己取消的数据
df = df[df['state'].str.match('user_canceled') != True]
print("去除用户自己取消后的数据量: {}".format(df.shape))

# 处理running_overdue 和 return_overdue 的逾期 的 check_result
df.loc[df['state'].str.contains('overdue') == True, 'check_result'] = 'FAILURE'
df['check_result'] = df['check_result'].map(lambda x: 1 if 'SUCCESS' in x else 0)

# df.to_csv(r'C:\Users\Administrator\iCloudDrive\蜜宝数据\蜜宝数据-已去除无用字段.csv', index=False)

# 处理detail_json
# detail_cols = ['strategySet', 'finalScore', 'success', 'result_desc', 'finalDecision']
# for col in detail_cols:
#     df[col] = df['detail_json'].map(lambda x: json.loads(x).get(col) if isinstance(x, str) else None)
#
# detail_cols = ['INFOANALYSIS', 'RENT']
# for col in detail_cols:
#     df[col] = df['result_desc'].map(lambda x: x.get(col) if isinstance(x, dict) else None)
# detail_cols = ['geotrueip_info', 'device_info', 'address_detect', 'geoip_info']
# for col in detail_cols:
#     df[col] = df['INFOANALYSIS'].map(lambda x: x.get(col) if isinstance(x, dict) else None)
# detail_cols = ['risk_items', 'final_score', 'final_decision']
# for col in detail_cols:
#     df[col] = df['RENT'].map(lambda x: x.get(col) if isinstance(x, dict) else None)
#
# df.drop(['result_desc', 'INFOANALYSIS', 'RENT'], axis=1, errors='ignore', inplace=True)
#
# cols = []
# for detail in df['RENT']:
#     if isinstance(detail, dict):
#         try:
#             cols.extend(list(detail.keys()))
#         except:
#             print(detail)
#             break
#
# cols = list(set(cols))
# print(cols)
#
# cols = []
# for detail in df['result_desc']:
#     if isinstance(detail, str):
#         try:
#             detail_dict = json.loads(detail)
#             cols.extend(list(detail_dict.keys()))
#         except:
#             print(detail)
#             # break
#
# cols = list(set(cols))
# print(cols)


# 特征处空值处理
# channel -随机处理
#  pay_type# ip# zmxy_score# card_id# contact# phone# provice# city# regoin# receive_address
# emergency_contact_name# phone_book# emergency_contact_phone# emergency_contact_relation# type.1# detail_json
# df.loc[df['discount'].isnull(), 'discount'] = 0
# df.loc[df['added_service'].isnull(), 'added_service'] = 0
# df.loc[df['first_pay'].isnull(), 'first_pay'] = 0


# df['card_id'].value_counts()
# len(df[df['card_id'].isnull()])
# for col in df.columns:
#     if len(df[df[col].isnull()]) != 0:
#         print(col)

# 处理芝麻信用分 '>600' 更改成600
row = 0
zmf = [0] * len(df)
xbf = [0] * len(df)
for x in df['zmxy_score']:
    # print(x, row)
    if isinstance(x, str):
        if '/' in x:
            score = x.split('/')
            xbf[row] = 0 if score[0] == '' else int(float(score[0]) / 10)
            zmf[row] = 0 if score[1] == '' else int(float(score[1]) / 100)
            # print(score, row)
        elif '>' in x:
            zmf[row] = 6
        else:
            score = float(x)
            if score <= 200:
                xbf[row] = int(score / 10)
            else:
                zmf[row] = int(score / 100)

    row += 1
df['zmf_score'] = zmf
df['xbf_score'] = xbf

# 根据身份证号增加性别和年龄 年龄的计算需根据订单创建日期计算
df['age'] = df['card_id'].map(lambda x: 2018 - int(x[6:10]))
df['sex'] = df['card_id'].map(lambda x: int(x[-2]) % 2)

# 处理phone_book, 只判断是否有phone book
df['phone_book'] = df['phone_book'].map(lambda x: 1 if isinstance(x, str) else 0)
# 处理phone_book, 只判断是否有phone book
df['phone'] = df['phone'].map(lambda x: x[0:3])
# 处理price, 大于3000000 赋值成3万
df['price'].where(df['price'] < 3000000, 3000000, inplace=True)
# 去除续租订单
df = df[df['releted'] == 0]
# 计算保险和意外险费用

# 所有空值赋值成0
df.fillna(value=0, inplace=True)

df.drop(labels=['zmxy_score', 'card_id', 'releted'], axis=1, inplace=True, errors='ignore')
# 暂时去除
features_drop = ['goods_name', 'goods_type',
                 'provice', 'city', 'regoin', 'type.1', 'detail_json',
                 'result', 'emergency_contact_phone',
                 'emergency_contact_relation',
                 'pay_type', 'merchant_id', 'channel', 'type', 'source',  # 影响比较小的元素
                 'ip', 'order_type', 'receive_address',  # 可能有用但还不知道怎么用
                 'state', 'cancel_reason', 'check_remark',]  # 丢弃相关数据
df.drop(labels=features_drop, axis=1, inplace=True, errors='ignore')
print("数据清理后的数据量: {}".format(df.shape))
SAVE_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", "mibaodata_ml.csv")
df.to_csv(SAVE_PATH, index=False)

exit(0)
# analyze data
def counter_scatter(data, showpic=True):
    vc = data.value_counts()
    print(vc)
    if (showpic):
        df_vc = pd.DataFrame({'value': vc.index, 'counts': vc.values})
        df_vc.plot(kind='scatter', x='value', y='counts', marker='.', alpha=0.4)


df.head()
df.info()
df.describe()
df.hist(bins=50, figsize=(20, 15))
counter_scatter(df['deposit'], False)
counter_scatter(df['phone_book'])
counter_scatter(df['price'] / 100)
plt.axis([0, 20000, 0, 4000])
df[['deposit', 'phone_book']].info()
# emergency_contact_phone, phone_book 这些数据只有7000个左右， 有缺失？？？？？
df.sort_values(by='price', inplace=True, ascending=False)
# # Discover and visualize the data to gain insights
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
# save_fig("housing_prices_scatterplot")

corr_matrix = df.corr()
corr_matrix["check_result"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["check_result", "freeze_money", "phone_book",
              "zmf_score"]
scatter_matrix(df[attributes], figsize=(12, 8))
# save_fig("scatter_matrix_plot")

df.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.axis([0, 16, 0, 550000])
# save_fig("income_vs_house_value_scatterplot")

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()

print("Missiong Complete!")
