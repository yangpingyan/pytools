# -*- coding: utf-8 -*-
"""
Created on Mon May  7 01:40:41 2018

@author: ypy
"""

import easytrader
import tushare as ts

print(easytrader.__version__)
print(ts.__version__)

print("Missioin Start")

try:
    print(user)
except:
    print("Initial easytrader")
#    user = easytrader.use('ths')
#    user.connect(r'C:\ths\xiadan.exe') # 类似 r'C:\htzqzyb2\xiadan.exe'
    user = easytrader.use('gj_client')
    user.prepare(user='39027298', password='616506', comm_password='920103')





df = ts.get_realtime_quotes('002413')
buy_price = float(df['ask'][0])+0.01
print(buy_price)

#print(user.today_trades)
#entrust_no = user.buy('002413', price=buy_price, amount=3000)

#print(entrust_no)

ret = user.cancel_entrust(entrust_no = '1766')
print(ret)


#user2 = easytrader.use('ths')
#user2.connect(r'C:\ths2\xiadan.exe') # 类似 r'C:\htzqzyb2\xiadan.exe'
#print(user2.balance)

#user.position
#user.buy('162411', price=0.55, amount=100)
#user.sell('162411', price=0.55, amount=100)
#user.cancel_entrust('buy/sell 获取的 entrust_no')
#user.today_trades