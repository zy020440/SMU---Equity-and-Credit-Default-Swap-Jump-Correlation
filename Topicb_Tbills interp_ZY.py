# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:13:54 2020

@author: Zhou Yan
"""

import numpy as np
import pandas as pd

def Interp(start, end, n, s): ### n=quarter number, s=Total No. quarters
    return start + (end-start)*n/s

TB = pd.read_csv('FED_SVENY.csv', index_col=0, parse_dates=True)
TB = TB.astype('float')
TB.insert(0, 'TB_0', 0)

#TB['TB_9mth'] = TB.apply(lambda x: Interp(x['TB_6mth'], x['TB_1y'], 2), axis=1)
#TB.insert(2, 'TB_9mth', TB.apply(lambda x: Interp(x['TB_6mth'], x['TB_1y'], 2, 4), axis=1))
cn = [['TB_3m', 'TB_6m', 'TB_9m'],['TB_1y3m', 'TB_1y6m', 'TB_1y9m'], ['TB_2y3m', 'TB_2y6m', 'TB_2y9m'],
      ['TB_3y3m', 'TB_3y6m', 'TB_3y9m'], ['TB_4y3m', 'TB_4y6m', 'TB_4y9m'], ['TB_5y3m', 'TB_5y6m', 'TB_5y9m'],
      ['TB_6y3m', 'TB_6y6m', 'TB_6y9m'], ['TB_7y3m', 'TB_7y6m', 'TB_7y9m'], ['TB_8y3m', 'TB_8y6m', 'TB_8y9m'],
      ['TB_9y3m', 'TB_9y6m', 'TB_9y9m'], ['TB_10y3m', 'TB_10y6m', 'TB_10y9m'], 
      ['TB_11y3m', 'TB_11y6m', 'TB_11y9m'], ['TB_12y3m', 'TB_12y6m', 'TB_12y9m'],
      ['TB_13y3m', 'TB_13y6m', 'TB_13y9m'], ['TB_14y3m', 'TB_14y6m', 'TB_14y9m'],
      ['TB_15y3m', 'TB_15y6m', 'TB_15y9m'], ['TB_16y3m', 'TB_16y6m', 'TB_16y9m'],
      ['TB_17y3m', 'TB_17y6m', 'TB_17y9m'], ['TB_18y3m', 'TB_18y6m', 'TB_18y9m'],
      ['TB_19y3m', 'TB_19y6m', 'TB_19y9m'], ['TB_20y3m', 'TB_20y6m', 'TB_20y9m'],
      ['TB_21y3m', 'TB_21y6m', 'TB_21y9m'], ['TB_22y3m', 'TB_22y6m', 'TB_22y9m'],
      ['TB_23y3m', 'TB_23y6m', 'TB_23y9m'], ['TB_24y3m', 'TB_24y6m', 'TB_24y9m'],
      ['TB_25y3m', 'TB_25y6m', 'TB_25y9m'], ['TB_26y3m', 'TB_26y6m', 'TB_26y9m'],
      ['TB_27y3m', 'TB_27y6m', 'TB_27y9m'], ['TB_28y3m', 'TB_28y6m', 'TB_28y9m'],
      ['TB_29y3m', 'TB_29y6m', 'TB_29y9m']]
lst = TB.columns.tolist()
cl = [1, 2, 3]
qt = [1, 2, 3]   ###quarter number

for i in range(len(cn)):
    for j in range(3):
        TB.insert(cl[j], cn[i][j], 1.1)
        TB[cn[i][j]] = TB.apply(lambda x: Interp(x[lst[i]], x[lst[i+1]], qt[j], 4), axis=1)
    
    for z in range(3):
        cl[z] = cl[z]+4

TB = TB.drop('TB_0', axis=1)
TB = TB.rename(columns = {'SVENY01': 'TB_1y', 'SVENY02': 'TB_2y', 'SVENY03': 'TB_3y', 'SVENY04': 'TB_4y', 
                          'SVENY05': 'TB_5y', 'SVENY06': 'TB_6y', 'SVENY07': 'TB_7y', 'SVENY08': 'TB_8y', 
                          'SVENY09': 'TB_9y', 'SVENY10': 'TB_10y', 'SVENY11': 'TB_11y', 'SVENY12': 'TB_12y', 
                          'SVENY13': 'TB_13y', 'SVENY14': 'TB_14y', 'SVENY15': 'TB_15y', 'SVENY16': 'TB_16y', 
                          'SVENY17': 'TB_17y', 'SVENY18': 'TB_18y', 'SVENY19': 'TB_19y', 'SVENY20': 'TB_20y', 
                          'SVENY21': 'TB_21y', 'SVENY22': 'TB_22y', 'SVENY23': 'TB_23y', 'SVENY24': 'TB_24y', 
                          'SVENY25': 'TB_25y', 'SVENY26': 'TB_26y', 'SVENY27': 'TB_27y', 'SVENY28': 'TB_28y', 
                          'SVENY29': 'TB_29y', 'SVENY30': 'TB_30y'})
TB
TB.to_csv(r"E:\Vicki's Class Schedule\SMU Class Schedule\QF608\CDSCOMP\TB.csv")



















