import pandas as pd
import numpy as np
import os
import datetime
from function import Equil1,Equil2,Equil3,Equil4,Equil5,Equil6,Equil7,Equil8,Equil9,Equil10,Equil11
from scipy.optimize import fsolve
# df=pd.read_csv("V5 CDS Composites-01Apr03.csv",index_col=0,header=0)
# z=df.loc[(df.Ticker.str.contains('THC|CHK|BBY|CVC'))&(df.Ccy=='USD')&(df.DocClause=='XR14')]
import CONFIG
# def import_stock_data(stock_code):
#     df=pd.read_csv(config.input_data_path+'/' + stock_code+ '.csv')
#     return df
def get_stock_code_list_in_one_dir(path):
    """
    从指定文件夹下，导入所有csv文件的文件名
    :param path:
    :return:
    """
    stock_list = []
    # 系统自带函数os.walk，用于遍历文件夹中的所有文件
    for root, dirs, files in os.walk(path):
        if files:  # 当files不为空的时候
            for f in files:
                if f.endswith('.csv'):
                    stock_list.append(f[:25])
    return stock_list
stock_code_list = get_stock_code_list_in_one_dir(CONFIG.input_data_path)
#k=[]


print(stock_code_list)
print(len(stock_code_list))


HR = []
all_lambda_data=pd.DataFrame()
#part1---------input and process data
for i in stock_code_list:
    dz=pd.read_csv(i,index_col=0,parse_dates=True)
    DF = pd.read_csv("E:\Vicki's Class Schedule\SMU Class Schedule\QF608\CDSCOMP\DF.csv", index_col=0, parse_dates=True)
    HRB=dz.dropna(subset=['Spread6m','Spread1y', 'Spread2y', 'Spread3y',
       'Spread4y', 'Spread5y', 'Spread7y', 'Spread10y', 'Spread15y',
       'Spread20y', 'Spread30y', 'Recovery', 'Rating6m', 'Rating1y',
       'Rating2y', 'Rating3y', 'Rating4y', 'Rating5y', 'Rating7y', 'Rating10y'],axis=0,how="any")#将所有的从6m---10y只要出现缺失值就删除

    HRB_r = HRB  ### dates in HRB but not in TBills
    HRB_r = HRB_r.drop(HRB_r.index[0:len(HRB.index)])
    for i in range(len(HRB.index)):
        if HRB.index[i] not in DF.index:
            pass
        else:
            HRB_r = HRB_r.append(HRB.iloc[i, :])

    lst = HRB_r.columns[7:19].tolist()
    HRB_r = HRB_r.iloc[:, range(7, 19)]

    for i in range(12):
        HRB_r[lst[i]] = HRB_r[lst[i]].str.strip('%').astype('float')/100

    DF_HRB = DF.loc[HRB_r.index.values, :]
    HRB_res = pd.concat([DF_HRB, HRB_r], axis=1)

#lambda1
    f = lambda s: fsolve(lambda x: Equil1(s[120], x, s[[0, 1]], s[131]), 0.5)[0]
    HRB_res['lambda1'] = HRB_res.apply(f, axis=1)


###lambda2

    f2 = lambda s: fsolve(lambda x: Equil2(s[121], s[132], x, s[[0, 1, 2, 3]], s[131]), 0.5)[0]
    HRB_res['lambda2'] = HRB_res.apply(f2, axis=1)

###lambda3

    f3 = lambda s: fsolve(lambda x: Equil3(s[122], s[132], s[133], x, s[list(range(8))], s[131]), 0.5)[0]
    HRB_res['lambda3'] = HRB_res.apply(f3, axis=1)
###lambda4


    f4 = lambda s: fsolve(lambda x: Equil4(s[123], s[132], s[133], s[134], x, s[list(range(12))], s[131]), 0.5)[0]
    HRB_res['lambda4'] = HRB_res.apply(f4, axis=1)

###lambda5


    f5 = lambda s: fsolve(lambda x: Equil5(s[124], s[132], s[133], s[134], s[135],
                                       x, s[list(range(16))], s[131]), 0.5)[0]
    HRB_res['lambda5'] = HRB_res.apply(f5, axis=1)

###lambda6


    f6 = lambda s: fsolve(lambda x: Equil6(s[125], s[132], s[133], s[134], s[135], s[136],
                                       x, s[list(range(20))], s[131]), 0.5)[0]
    HRB_res['lambda6'] = HRB_res.apply(f6, axis=1)

###lambda7


    f7 = lambda s: fsolve(lambda x: Equil7(s[126], s[132], s[133], s[134], s[135], s[136], s[137],
                                       x, s[list(range(28))], s[131]), 0.5)[0]
    HRB_res['lambda7'] = HRB_res.apply(f7, axis=1)

###lambda8


    f8 = lambda s: fsolve(lambda x: Equil8(s[127], s[132], s[133], s[134], s[135], s[136], s[137], s[138],
                                       x, s[list(range(40))], s[131]), 0.5)[0]
    HRB_res['lambda8'] = HRB_res.apply(f8, axis=1)

###lambda9


    f9 = lambda s: fsolve(lambda x: Equil9(s[128], s[132], s[133], s[134], s[135], s[136], s[137], s[138], s[139],
                                       x, s[list(range(60))], s[131]), 0.5)[0]
    HRB_res['lambda9'] = HRB_res.apply(f9, axis=1)

###lambda10

    f10 = lambda s: fsolve(lambda x: Equil10(s[129], s[132], s[133], s[134], s[135], s[136], s[137], s[138],
                                         s[139], s[140], x, s[list(range(80))], s[131]), 0.5)[0]
    HRB_res['lambda10'] = HRB_res.apply(f10, axis=1)

###lambda11

    f11 = lambda s: fsolve(lambda x: Equil11(s[130], s[132], s[133], s[134], s[135], s[136], s[137], s[138],
                                         s[139], s[140], s[141], x, s[list(range(120))], s[131]), 0.2)[0]
    HRB_res['lambda11'] = HRB_res.apply(f11, axis=1)

###export   
    #all_lambda_data = all_lambda_data.append(HRB_res[['lambda1', 'lambda2', 'lambda3', 'lambda4', 
    #                                              'lambda5', 'lambda6', 'lambda7','lambda8', 
    #                                              'lambda9', 'lambda10','lambda11']])
    all_lambda_data = HRB_res[['lambda1', 'lambda2', 'lambda3', 'lambda4', 
                                                  'lambda5', 'lambda6', 'lambda7','lambda8', 
                                                  'lambda9', 'lambda10','lambda11']]

    HR.append(all_lambda_data)
    
for i in range(len(HR)):
    HR[i].to_csv(os.path.join(r'HR', str(stock_code_list[i])))

