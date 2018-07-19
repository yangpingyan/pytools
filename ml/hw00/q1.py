#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pandas as pd


index = 1
input_file = "hw0_data.dat"

df = pd.read_table(input_file, sep=' ', header=None)
df = df.sort_values(1)
print(df)


with open("out.txt", "w") as f:
    for val in df[1]:
        print(val)
        f.write(str(val))
print("Mission Complete!")




