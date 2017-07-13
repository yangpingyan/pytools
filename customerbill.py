#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#安装mysqlclient
brew uninstall mysql-connector-c
brew install mysql
pip install mysqlclient

Created on Wed Jul 12 19:52:44 2017

@author: zhanghua
"""
import json
import MySQLdb
from sshtunnel import SSHTunnelForwarder  
import pandas as pd

def file2dict(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
    
print("MISSION START") 

rdsinfo = file2dict('pannardspw.json')
sshhost = rdsinfo['sshhost']
sshpassword = rdsinfo['sshpassword']
sshbindaddress = rdsinfo['sshbindaddress']
rdsuser = rdsinfo['rdsuser']
rdspassword = rdsinfo['rdspassword']
rdsdb = rdsinfo['rdsdb']


server = SSHTunnelForwarder((sshhost, 22),    #B机器的配置  
                            ssh_username="root", 
                            ssh_password=sshpassword,
                            remote_bind_address=(sshbindaddress, 3388)) #A机器的配置  
    
server.start() #start ssh sever 
    
conn = MySQLdb.connect(host='127.0.0.1',              #此处必须是是127.0.0.1  
                       port=server.local_bind_port,  
                       user=rdsuser,  
                       passwd=rdspassword,  
                       db=rdsdb)  
cursor = conn.cursor()

sql = """SELECT p.name, p.type, p.status, d.buying_money, d.buying_rate, 
    d.buying_limit_num_show, d.buying_type, d.status
    FROM fy_product_detail d 
    LEFT JOIN fy_product p ON p.`id` = d.`product_id` 
    LEFT JOIN fy_user u ON u.id = d.`user_id`
    WHERE u.`real_name` = 'tt21'
    and d.status = 2;"""   

df = pd.read_sql(sql,conn)

#产品名称	产品类型	项目状态	认购金额	项目期限	利率/层级	理财师	客户姓名
#df = df[['name','type', 'status']]
print(df)
     


cursor.close() 
conn.close() 
server.stop()

print("MISSION COMPLETE")
