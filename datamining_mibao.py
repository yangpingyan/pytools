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
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import os


# Function to calculate missing values by column# Funct
from sklearn.preprocessing import LabelEncoder


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# to make output display better
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 1000)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.sans-serif'] = ['Simhei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# read large csv file
csv.field_size_limit(100000000)

# Datasets info
PROJECT_ROOT_DIR = os.getcwd()
DATA_ID = "学校数据.csv"
DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", DATA_ID)
df_alldata = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
print("初始数据量: {}".format(df_alldata.shape))
df = df_alldata.dropna(axis=1, how='all')

# 处理身份证号
df['card_id'] = df['card_id'].map(lambda x: x.replace(x[10:16], '******') if isinstance(x, str) else x)

# 取可能有用的数据
# 目前discount字段是用户下单通过后才生成的，无法使用。
# 建议保存用户下单时的优惠金额，因为违约用户大概率是不计较优惠金额，
features = ['create_time', 'goods_name', 'cost', 'discount', 'pay_num', 'added_service', 'first_pay', 'channel',
            'pay_type', 'merchant_id', 'goods_type', 'lease_term', 'daily_rent', 'accident_insurance', 'type',
            'freeze_money', 'ip', 'releted', 'order_type', 'delivery_way', 'source', 'disposable_payment_discount',
            'disposable_payment_enabled', 'lease_num', 'original_daily_rent', 'deposit', 'zmxy_score', 'card_id',
            'contact', 'phone', 'provice', 'city', 'regoin', 'receive_address', 'emergency_contact_name', 'phone_book',
            'emergency_contact_phone', 'emergency_contact_relation', 'type.1', 'detail_json', 'price', 'old_level']
result = ['state', 'cancel_reason', 'check_result', 'check_remark', 'result']
df = df[result + features]
print("筛选出所有可能有用特征后的数据量: {}".format(df.shape))

df.info()
df.describe()
# Missing values statistics
missing_values = missing_values_table(df)
missing_values.head(20)
# Number of each type of column
df.dtypes.value_counts()
# Number of unique classes in each object column
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

df.describe(include=['O'])



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

# 去除身份证重复的订单：
df.drop_duplicates(subset=['card_id'], keep='last', inplace=True)
print("去除身份证重复的订单后的数据量: {}".format(df.shape))

# 处理running_overdue 和 return_overdue 的逾期 的 check_result
df.loc[df['state'].str.contains('overdue') == True, 'check_result'] = 'FAILURE'
df['check_result'] = df['check_result'].apply(lambda x: 1 if 'SUCCESS' in x else 0)

# 有phone_book的赋值成1， 空的赋值成0(空值最后统一赋值）
df['phone_book'][df['phone_book'].notnull()] = 1

# 根据create_time 按时间段分类
df['create_hour'] = df['create_time'].map(lambda x: int(x[-8:-6]))
df['create_time_cat'] = df['create_hour'].map(lambda x: 0 if 0 < x < 7 else 1)

# 有emergency_contact_phone的赋值成1， 空的赋值成0(空值最后统一赋值）
df['emergency_contact_phone'][df['emergency_contact_phone'].notnull()] = 1


# 服务费first_pay = 租赁天数*每日租金 +保险和增值费。（短租）
#     first_pay = 每期天数*每日租金 + 保险和增值费。 （长租）
#     cost = first_pay （短租），
#     cost = 总租赁天数*每日租金（长租）
# df['pay'] = df['first_pay'] * df['lease_term']
# df[['pay', 'first_pay', 'lease_term', 'daily_rent', 'pay_num', 'accident_insurance']]
# check_pass = pd.DataFrame({'pass': df[df['check_result'] == 1]['create_hour'].value_counts()})
# check_all = pd.DataFrame({'all': df['create_hour'].value_counts()})
# check_hour = check_pass.merge(check_all, how='outer', left_index=True, right_index=True)
# check_hour['pass_rate'] = check_hour['pass'] / check_hour['all'] * 100
# plt.plot(check_hour['pass_rate'], 'yo')


# 处理detail_json
df['result'] = df['result'].map(lambda x: x.upper() if isinstance(x, str) else 'NODATA')
df['result'][df['result'].str.match('ACCEPT')] = 'PASS'

# 数据调试代码
val = 'result'
df[['check_result', val]].info()
df[val].value_counts()
df[df[val].isnull()]
df.sort_values(by=[val], inplace=True)
check_pass = pd.DataFrame({'pass': df[df['check_result'] == 1][val].value_counts()})
check_reject = pd.DataFrame({'reject': df[df['check_result'] == 0][val].value_counts()})
check_all = pd.DataFrame({'all': df[val].value_counts()})
check_df = check_pass.merge(check_all, how='outer', left_index=True, right_index=True)
check_df = check_df.merge(check_reject, how='outer', left_index=True, right_index=True)
check_df['pass_rate'] = check_df['pass'] / check_df['all'] * 100
check_df['reject_rate'] = check_df['reject'] / check_df['all'] * 100
plt.plot(check_df['pass_rate'], 'gs')
plt.plot(check_df['reject_rate'], 'ro')
# 展开detail_json中所有的字典
def expand_dict(dict_in):
    dict_out = dict()
    for k, v in dict_in.items():
        # print(k, v)
        if isinstance(v, dict):
            dict_out.update(expand_dict(v))
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    dict_out.update(expand_dict(d))
                else:
                    dict_out[k] = v
        else:
            dict_out[k] = v
    return dict_out


dict_out = dict()
for val in df['detail_json']:
    if (isinstance(val, str)):
        dict_out.update(expand_dict(json.loads(val)))

detail_cols = ['success', 'final_score', 'score', 'decision', 'risk_name', 'hit_type_display_name',
               'fraud_type_display_name', 'evidence_time', 'risk_level', 'fraud_type', 'value', 'type', 'data',
               'detail', 'count', 'dimension', 'platform_count', 'final_decision', 'discredit_times', 'overdue_time',
               'overdue_amount_range', 'fuzzy_id_number', 'fuzzy_name', 'overdue_day_range', 'high_risk_areas',
               'hit_list_datas', 'overdue_count', 'execute_subject', 'execute_court', 'case_code', 'executed_name',
               'case_date', 'evidence_court', 'execute_status', 'term_duty', 'gender', 'carry_out', 'execute_code',
               'province', 'specific_circumstances', 'age', 'finalDecision', 'finalScore', 'memo', 'ruleId', 'ruleName',
               'template', 'riskType', 'strategyMode', 'strategyName', 'strategyScore', 'firstType', 'grade',
               'secondType', 'name', 'rejectValue', 'reviewValue', 'true_ip_address', 'mobile_address',
               'id_card_address', 'isp', 'latitude', 'position', 'longitude', 'error', 'proxyProtocol', 'port',
               'proxyType']
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
            xbf[row] = 0 if score[0] == '' else (float(score[0]))
            zmf[row] = 0 if score[1] == '' else (float(score[1]))
            # print(score, row)
        elif '>' in x:
            zmf[row] = 600
        else:
            score = float(x)
            if score <= 200:
                xbf[row] = (score)
            else:
                zmf[row] = (score)

    row += 1
df['zmf_score'] = zmf
df['xbf_score'] = xbf

# 根据身份证号增加性别和年龄 年龄的计算需根据订单创建日期计算
df['age'] = df['card_id'].map(lambda x: 2018 - int(x[6:10]))
df['sex'] = df['card_id'].map(lambda x: int(x[-2]) % 2)

# df.sort_values(by=['merchant_id'], inplace=True)

# df.dropna(subset=['zmf_score', 'xbf_score'], inplace=True)
# df = df[df['xbf_score'] > 0]
# df = df[df['zmf_score'] > 0]
df.fillna(value=0, inplace=True)
print("保存的数据量: {}".format(df.shape))
df.to_csv(os.path.join(PROJECT_ROOT_DIR, "datasets", "mibaodata_ml.csv"), index=False)
df.head()
df.info()
df.describe()
df.hist(bins=50, figsize=(20, 15))

# # Discover and visualize the data to gain insights
df.plot(kind="scatter", x="zmf_score", y="xbf_score", alpha=0.4,
        s=df["check_result"], label="check_result", figsize=(10, 7),
        c="check_result", cmap=plt.get_cmap("jet"), colorbar=True,
        sharex=False)
plt.legend()

corr_matrix = df.corr()
corr_matrix["check_result"].sort_values(ascending=False)
#
# from pandas.plotting import scatter_matrix
#
# attributes = ["median_house_value", "median_income", "total_rooms",
#               "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# # save_fig("scatter_matrix_plot")
#
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.axis([0, 16, 0, 550000])
# # save_fig("income_vs_house_value_scatterplot")
#
# housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
# housing["population_per_household"] = housing["population"] / housing["households"]
#
# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)
#
# housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
# plt.axis([0, 5, 0, 520000])
# plt.show()

print("Missiong Complete!")
