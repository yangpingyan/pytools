#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2018/8/2 16:15 
# @Author : yangpingyan@gmail.com

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import re


csv.field_size_limit(100000000)

df_alldata = pd.read_csv(r"C:\Users\Administrator\iCloudDrive\蜜宝数据\学校用数据.csv", encoding='utf-8', engine='python')
print("原始数据量: {}".format(df_alldata.shape))
df = df_alldata.dropna(axis=1, how='all')
print("去除所有特征为空后的数据量: {}".format(df.shape))
# features_drop = ['user_name', 'phone' ]
# df.drop(columns=features_drop, axis=1, inplace=True)
# print("去除无效特征后的数据量: {}".format(df.shape))

# 处理身份证号
df['card_id'][0:20].apply(lambda x: x.replace(x[10:16], '******') if isinstance(x, str) else x )



# 取可能有用的数据
features = ['cost', 'discount', 'pay',
            'installment', 'pay_num', 'added_service', 'first_pay', 'full', 'billing_method',
            'channel', 'pay_type', 'goods_type', 'cash_pledge', 'lease_term', 'commented', 'daily_rent',
            'accident_insurance', 'freeze_money', 'ip', 'pid', 'releted', 'order_type', 'delivery_way',
            'source', 'disposable_payment_discount', 'disposable_payment_enabled',
            'lease_num', 'original_daily_rent', 'deposit', 'zmxy_score', 'card_id', 'contact',
            'phone', 'provice', 'city', 'regoin', 'receive_address', 'freight',
            'emergency_contact_name', 'phone_book', 'emergency_contact_phone', 'emergency_contact_relation',
            'type', 'remark', 'result', 'detail_json', 'joke']
result = ['cancel_reason', 'credit_check_result', 'check_result', 'check_remark', 'finished_state']
df = df[result + features]
print("筛选出所有可能有用特征后的数据量: {}".format(df.shape))



# 取有审核结果的数据
df = df[df['check_result'].str.contains('SUCCESS|FAILURE')]
print("去除机审通过但用户取消后的数据量: {}".format(df.shape))
# 去除只有唯一值的列
for col in df.columns:
    if len(df[col].unique()) <= 1:
        df.drop(col, axis=1, errors='ignore', inplace=True)
print("去除特征值中只有唯一值后的数据量: {}".format(df.shape))

# 特征处理
df['check_result'] = df['check_result'].apply(lambda x: 1 if 'SUCCESS' in x else 0)

# 处理芝麻信用分
df['zmxy_score'] = df['zmxy_score'].apply(lambda x: 599 if '>' in x else round(float(x)))
df.sort_values(by=['zmxy_score'], inplace=True)

check_counts = df['check_result'].value_counts()
print("所有数据中：审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[df['zmxy_score'] == 599]['check_result'].value_counts()
print("未知大于600的芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[(df['zmxy_score']>=600) & (df['zmxy_score']<700)]['check_result'].value_counts()
print("6XX芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[(df['zmxy_score']>=700) & (df['zmxy_score']<800)]['check_result'].value_counts()
print("7XX芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[df['zmxy_score'] >= 800]['check_result'].value_counts()
print("大于800芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
#
# avail_zmxy_df = df[df['zmxy_score'] > 599]
# avail_zmxy_s = pd.Series(avail_zmxy_df['zmxy_score'].value_counts())
# failure_zmxy_df = avail_zmxy_df[avail_zmxy_df['check_result'] == 0]
# failure_zmxy_s = pd.Series(failure_zmxy_df['zmxy_score'].value_counts())
# s_zmxy_df = avail_zmxy_df[avail_zmxy_df['check_result'] == 1]
# s_zmxy_s = pd.Series(s_zmxy_df['zmxy_score'].value_counts())
# zmxy_view_df = pd.DataFrame({'avail_nums': avail_zmxy_s, 'fail_nums': failure_zmxy_s, 'success_nums': s_zmxy_s})
#
# # 绘图
# fig, ax = plt.subplots()
# bar_width = 0.35
# opacity = 0.4
#
# rects1 = ax.bar(avail_zmxy_s.index, avail_zmxy_s.values, bar_width,
#                 alpha=opacity, color='b',
#                 label='Men')
#
# rects2 = ax.bar(failure_zmxy_s.index + bar_width, failure_zmxy_s.values, bar_width,
#                 alpha=opacity, color='r',
#                 label='Women')
#
# ax.set_xlabel('Group')
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(avail_zmxy_s.index, + bar_width / 2)
# ax.legend()
#
# fig.tight_layout()
# plt.show()
