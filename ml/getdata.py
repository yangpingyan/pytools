#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2018/8/2 16:15 
# @Author : yangpingyan@gmail.com
import catch as catch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import re

csv.field_size_limit(100000000)

df_alldata = pd.read_csv(r"C:\Users\Administrator\iCloudDrive\蜜宝数据\学校所需数据2.csv", encoding='utf-8', engine='python')
print("原始数据量: {}".format(df_alldata.shape))
df = df_alldata.dropna(axis=1, how='all')
print("去除所有特征为空后的数据量: {}".format(df.shape))
# features_drop = ['user_name', 'phone' ]
# df.drop(columns=features_drop, axis=1, inplace=True)
# print("去除无效特征后的数据量: {}".format(df.shape))

# 处理身份证号
df['card_id'] = df['card_id'].apply(lambda x: x.replace(x[10:16], '******') if isinstance(x, str) else x)

# 取可能有用的数据
features = ['goods_name', 'cost', 'discount', 'installment', 'pay_num', 'added_service', 'first_pay', 'full', 'channel',
            'pay_type', 'merchant_id', 'goods_type', 'lease_term', 'daily_rent', 'accident_insurance', 'type',
            'freeze_money', 'ip', 'releted', 'order_type', 'delivery_way', 'source', 'disposable_payment_discount',
            'disposable_payment_enabled', 'lease_num', 'original_daily_rent', 'deposit', 'zmxy_score', 'card_id',
            'contact', 'phone', 'provice', 'city', 'regoin', 'receive_address', 'emergency_contact_name', 'phone_book',
            'emergency_contact_phone', 'emergency_contact_relation', 'type.1', 'detail_json']
result = ['state', 'cancel_reason', 'check_result', 'check_remark', 'result']
df = df[result + features]
print("筛选出所有可能有用特征后的数据量: {}".format(df.shape))

# df.to_csv(r'C:\Users\Administrator\iCloudDrive\蜜宝数据\蜜宝数据-已去除无用字段.csv', index=False)

# 取有审核结果的数据
df = df[df['check_result'].str.contains('SUCCESS|FAILURE', na=False)]
print("去除机审通过但用户取消后的数据量: {}".format(df.shape))
# 去除只有唯一值的列
cols = []
for col in df.columns:
    if len(df[col].unique()) <= 1:
        cols.append(col)
df.drop(cols, axis=1, errors='ignore', inplace=True)
print("去除特征值中只有唯一值后的数据量: {}".format(df.shape))

# 去除测试数据
df = df[df['cancel_reason'].str.contains('测试') != True]
df = df[df['check_remark'].str.contains('测试') != True]
print("去除测试数据后的数据量: {}".format(df.shape))

# 去掉用户自己取消的数据
df = df[df['state'].str.contains('user_canceled') != True]
print("去除用户自己取消后的数据量: {}".format(df.shape))

# for col in df.columns:
#     try:
#         print(col, len(df[df[col].str.contains('测试') == True]))
#     except:
#         pass

# 特征处理
# df['check_result'] = df['check_result'].apply(lambda x: 1 if 'SUCCESS' in x else 0)

# 处理芝麻信用分
# df['jdxb_score'] = df['zmxy_score'].apply(lambda x: float(x.split('/')[0]) if isinstance(x, str) and '/' in x else None)

row = 0
zmf = [None] * len(df)
xbf = [None] * len(df)
for x in df['zmxy_score']:
    # print(x, row)
    if isinstance(x, str):
        if '/' in x:
            score = x.split('/')
            xbf[row] = None if score[0] == '' else float(score[0])
            zmf[row] = None if score[1] == '' else float(score[1])
            # print(score, row)
        elif '>' in x:
            zmf[row] = 600
        else:
            score = float(x)
            if score <= 200:
                xbf[row] = score
            else:
                zmf[row] = score

    row += 1
df['zmf_score'] = zmf
df['xbf_score'] = xbf

check_counts = df['check_result'].value_counts()
print("所有数据中：审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[df['zmf_score'] < 600]['check_result'].value_counts()
print("小于600的芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                                   check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[(df['zmf_score'] >= 600) & (df['zmf_score'] < 700)]['check_result'].value_counts()
print("6XX芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                                check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[(df['zmf_score'] >= 700) & (df['zmf_score'] < 800)]['check_result'].value_counts()
print("7XX芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                                check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[df['zmf_score'] >= 800]['check_result'].value_counts()
print("大于800芝麻分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                                  check_counts[1] / (check_counts[0] + check_counts[1])))

check_counts = df[df['xbf_score'] < 60]['check_result'].value_counts()
print("小于60的小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                                  check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[(df['xbf_score'] >= 60) & (df['xbf_score'] < 70)]['check_result'].value_counts()
print("6X小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[(df['xbf_score'] >= 70) & (df['xbf_score'] < 80)]['check_result'].value_counts()
print("7X小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[(df['xbf_score'] >= 80) & (df['xbf_score'] < 90)]['check_result'].value_counts()
print("8X小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[(df['xbf_score'] >= 90) & (df['xbf_score'] < 100)]['check_result'].value_counts()
print("9X小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                               check_counts[1] / (check_counts[0] + check_counts[1])))
check_counts = df[df['xbf_score'] >= 100]['check_result'].value_counts()
print("大于100小白分中审核拒绝{}个，审核通过{}个，通过率{:.2f}".format(check_counts[0], check_counts[1],
                                                  check_counts[1] / (check_counts[0] + check_counts[1])))

df.sort_values(by=['state', 'cancel_reason'], inplace=True)

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
