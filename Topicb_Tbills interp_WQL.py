import numpy as np
import pandas as pd
from scipy import interpolate


#inteporate the rate

db=pd.read_csv("bill.csv",index_col=0,header=0)
db.insert(0,'0',0)
print(db)
x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
z=db.index
all_data=[]
for i in range(len(db.index)):
    y=db.loc[db.index[i],:].values
    B_function = interpolate.interp1d(x,y)
    B_quarterly_array = B_function(np.arange(0,30+0.25, 0.25))
    all_data.append(B_quarterly_array)
print(all_data)
df=pd.DataFrame(all_data,index=db.index,columns=np.arange(0,30+0.25, 0.25))
print(df)
df.to_csv('/Users/wangqinglin/Desktop/project文件/kk.csv')
