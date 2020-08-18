# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:39:26 2020

@author: Zhou Yan
"""


import numpy as np
import pandas as pd

TB = pd.read_csv('TB.csv', index_col=0, parse_dates=True)

def B(r, t, n): # t=quarterly payment   n=quarter number
    if 0<n<4:
        return 1/(1+n*r)
    else:
        return 1/((1+r)**(n*t))

TB = TB/100
DF = TB
lst = TB.columns.tolist()
for i in range(120):
    DF[lst[i]] = DF.apply(lambda x: B(x[lst[i]], 0.25, i+1), axis=1)

DF.to_csv(r"E:\Vicki's Class Schedule\SMU Class Schedule\QF608\CDSCOMP\DF.csv")
