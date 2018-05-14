# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:00:43 2018

@author: ypy
"""

import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='myapp.log',
                filemode='w')

logging.basicConfig(level = logging.DEBUG)
logging.debug("logging debug message")
logging.info("logging info message")
logging.warning("logging warning message")