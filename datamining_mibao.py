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

# to make output display better
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
#read large csv file
csv.field_size_limit(100000000)

# Datasets info
PROJECT_ROOT_DIR = os.getcwd()
DATA_ID = "学校测试所需数据.csv"
DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", DATA_ID)

df_alldata = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
df = df_alldata.dropna(axis=1, how='all')
df.info()
df.head()


# 处理身份证号
df['card_id'] = df['card_id'].apply(lambda x: x.replace(x[10:16], '******') if isinstance(x, str) else x)

# 取可能有用的数据
features = ['create_time', 'goods_name', 'cost', 'discount', 'pay_num', 'added_service', 'first_pay', 'channel',
            'pay_type', 'merchant_id', 'goods_type', 'lease_term', 'daily_rent', 'accident_insurance', 'type',
            'freeze_money', 'ip', 'releted', 'order_type', 'delivery_way', 'source', 'disposable_payment_discount',
            'disposable_payment_enabled', 'lease_num', 'original_daily_rent', 'deposit', 'zmxy_score', 'card_id',
            'contact', 'phone', 'provice', 'city', 'regoin', 'receive_address', 'emergency_contact_name', 'phone_book',
            'emergency_contact_phone', 'emergency_contact_relation', 'type.1', 'detail_json', 'price', 'old_level']
result = ['state', 'cancel_reason', 'check_result', 'check_remark', 'result']
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
df['check_result'] = df['check_result'].apply(lambda x: 1 if 'SUCCESS' in x else 0)

# df.to_csv(r'C:\Users\Administrator\iCloudDrive\蜜宝数据\蜜宝数据-已去除无用字段.csv', index=False)

# 处理detail_json
# detail_cols = ['strategySet', 'finalScore', 'success', 'result_desc', 'finalDecision']
# for col in detail_cols:
#     df[col] = df['detail_json'].apply(lambda x: json.loads(x).get(col) if isinstance(x, str) else None)
#
# detail_cols = ['INFOANALYSIS', 'RENT']
# for col in detail_cols:
#     df[col] = df['result_desc'].apply(lambda x: x.get(col) if isinstance(x, dict) else None)
# detail_cols = ['geotrueip_info', 'device_info', 'address_detect', 'geoip_info']
# for col in detail_cols:
#     df[col] = df['INFOANALYSIS'].apply(lambda x: x.get(col) if isinstance(x, dict) else None)
# detail_cols = ['risk_items', 'final_score', 'final_decision']
# for col in detail_cols:
#     df[col] = df['RENT'].apply(lambda x: x.get(col) if isinstance(x, dict) else None)
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

#
# check_counts = df['check_result'].value_counts()
# print("所有数据中：审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[df['zmf_score'] < 600]['check_result'].value_counts()
# print("小于600的芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                    check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[(df['zmf_score'] >= 600) & (df['zmf_score'] < 700)]['check_result'].value_counts()
# print("6XX芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                 check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[(df['zmf_score'] >= 700) & (df['zmf_score'] < 800)]['check_result'].value_counts()
# print("7XX芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                 check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[df['zmf_score'] >= 800]['check_result'].value_counts()
# print("大于800芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                   check_counts[1] / (check_counts[0] + check_counts[1])))
#
# check_counts = df[df['xbf_score'] < 60]['check_result'].value_counts()
# print("小于60小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                  check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[(df['xbf_score'] >= 60) & (df['xbf_score'] < 70)]['check_result'].value_counts()
# print("6X小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[(df['xbf_score'] >= 70) & (df['xbf_score'] < 80)]['check_result'].value_counts()
# print("7X小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[(df['xbf_score'] >= 80) & (df['xbf_score'] < 90)]['check_result'].value_counts()
# print("8X小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[(df['xbf_score'] >= 90) & (df['xbf_score'] < 100)]['check_result'].value_counts()
# print("9X小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                check_counts[1] / (check_counts[0] + check_counts[1])))
# check_counts = df[df['xbf_score'] >= 100]['check_result'].value_counts()
# print("大于100小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
#                                                   check_counts[1] / (check_counts[0] + check_counts[1])))

# df.sort_values(by=['merchant_id'], inplace=True)

# df.dropna(subset=['zmf_score', 'xbf_score'], inplace=True)
# df = df[df['xbf_score']>0]
df.to_csv("mibaodata_ml.csv", index=False)


df.describe()
df.hist(bins=50, figsize=(20, 15))
save_fig("attribute_histogram_plots")
# plt.show()

# # Discover and visualize the data to gain insights
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
# save_fig("housing_prices_scatterplot")

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# save_fig("scatter_matrix_plot")

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
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
