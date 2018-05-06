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

def getAllExcelFiles(file_dir): 
    L = []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:
            if os.path.splitext(file)[1] == '.xlsx':
                L.append(os.path.join(root, file))
    return L

files = getAllExcelFiles('.\customers')
print(files)

all_df = pd.DataFrame()
for excelfile in files:
    print(excelfile)
       
    df_file = pd.read_excel(excelfile, header=1)
    df_file.rename(columns={'客户':'客户姓名', '身份证号':'身份证号/营业执照号', '身份证号码/营业执照号码':'身份证号/营业执照号'}, inplace = True)
    print(df_file.columns)

    all_df = pd.concat([all_df, df_file], join='outer', axis=0, ignore_index=True)
  

all_df = all_df[['客户姓名', '身份证号/营业执照号']]
print(all_df)
#all_df.to_excel('allcustomers.xlsx')

print("mission complete")


