# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 14:18:11 2017

@author: yangp
"""


'''
Created on 2017-4-6

@author: James
'''
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from openpyxl import Workbook
from openpyxl import load_workbook
from datetime import datetime
import pandas as pd
from docx.oxml.ns import qn
import os
import string
import re

def getAllExcelFiles(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xlsx' \
                and file.startswith('1') is False \
                and file.startswith('.') is False \
                and file.startswith('~') is False:
                L.append(os.path.join(root, file))
    return L

if __name__ == '__main__':
    files = getAllExcelFiles('X:\股票账户统计')
    print("统计的账户个数是{}个".format(len(files)))

#    print(files)

    all_df = pd.DataFrame()
    for excelfile in files:
<<<<<<< HEAD

        df_file = pd.read_excel(excelfile, sheet_name='交易记录', header=0)
        df_file = df_file.dropna(how='all')
        if(df_file.empty):
            continue

=======
        df_file = pd.read_excel(excelfile, sheet_name='交易记录', header=0)
        df_file = df_file.dropna(how='all')
        if(df_file.empty):
            continue

>>>>>>> parent of a03d006... done
        df_file.rename(columns={'委托类别':'操作', '买入标志':'操作', \
                                '买卖标志':'操作', '成交均价':'成交价格', \
                                '发生金额':'成交金额',}, inplace = True)
        df_file = df_file[['成交日期', '证券名称', '操作', '成交数量', '成交价格', '成交金额']]

        df_file = df_file[df_file['证券名称'].str.contains('雷科防务')]
        account_name = re.findall(r"[\u4e00-\u9fa5]{1,}-[\u4e00-\u9fa5]{1,}", excelfile)
        df_file['账户'] = "".join(account_name)

        all_df = pd.concat([all_df, df_file], join='outer', axis=0, ignore_index=True)


    all_df['成交日期'] = [x if isinstance(x,str) else "%d"%x for x in all_df['成交日期']]
#    print(all_df)

    #取得合计持股数，与实际持股数核对以判断取得全部数据
    buy_df = all_df[all_df['操作'].str.contains('买入')]
    buy_amount = buy_df['成交数量'].sum()
    sell_df = all_df[all_df['操作'].str.contains('卖出')]
    sell_amount = sell_df['成交数量'].sum()
    print("总持股合计={:,.0f}".format(buy_amount-sell_amount))

    #按日期取得交易数据
<<<<<<< HEAD
    date_str = '20180507'
=======
    date_str = '20180509'
>>>>>>> parent of a03d006... done
    date_df = all_df[all_df['成交日期'].str.contains(date_str)]
    buy_df = date_df[date_df['操作'].str.contains('买入') ]
    buy_amount = buy_df['成交数量'].sum()
    sell_df = date_df[date_df['操作'].str.contains('卖出') ]
    sell_amount = sell_df['成交数量'].sum()
    print("{}买入股数合计={:,.0f}".format(date_str,buy_amount))
    print("{}卖出股数合计={:,.0f}".format(date_str,sell_amount))
<<<<<<< HEAD
    print("{}持股合计={:,.0f}".format(date_str,(buy_amount - sell_amount)))
=======
    print("{}持股合计={:,.0f}".format(date_str,buy_amount-sell_amount))
>>>>>>> parent of a03d006... done
#    print(date_df)

    print("mission complete")











