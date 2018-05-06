# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:44:30 2018

@author: yangp
"""

import pandas as pd
import numpy as np


from datetime import datetime
import operator
import math
import matplotlib.pyplot as plt

import itertools

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

c = np.where(a == b)
print(c)
print("Mission Complete")