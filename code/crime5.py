# final 이전 최종 테스트3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


crime = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/GEOID2.csv')
census = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/census.csv')
list(crime)

crime.dtypes
census.dtypes
# 0 조정
crime = pd.merge(crime,census, how = 'left', on = 'GEOID')
list(crime)
crime = crime.drop(['Unnamed: 0','Unnamed: 0.1','Lat','Long','Location','Year','Month','location'], axis = 1)
crime.dtypes

del_list = [
    'P0040001',
 'P0040005',
 'P0040006',
 'P0040002',
 'P0040008',
 'P0040007',
 'P0040009',
 'P0040010',
 'P0040011',
 'P0050001',
 'P0050002',
 'P0050003',
 'P0050004',
 'P0050005',
 'P0050006',
 'P0050007',
 'P0050008',
 'P0050009',
 'P0050010',
 'H0010001',
 'H0010002',
 'H0010003',]

crime = crime.drop(del_list,axis = 1)

# target value 생성
region = list(set(list(crime['GEOID'])))
crime['final_count'] = 0
date_list = list(set(list(crime['Date'])))

count = 0
for r in region:
    for date in date_list:
        is_date = crime['Date'] == date
        is_problem = crime['UCR_PART_Part One']
        is_center = crime['GEOID'] == r
        index_list = crime[is_date & is_problem & is_center].index
        number = len(index_list)
        for n in index_list:
            crime['final_count'][n] = number
            
        count+=1
        print(count)

# save point2
crime.to_csv('final_merge2_crime.csv')

# vif 체크
def vif(df1_names,df1):
    reg = LinearRegression()
    
    vif_name = []
    vif_arr = []
    
    for name in df1_names:
        templist = df1_names[:]
        templist.remove(name)
        p1 = templist
        p2 = name
        
        reg.fit(df1[p1], df1[p2])
        vif = 1/(1-reg.score(df1[p1], df1[p2]))
        vif_name.append(name)
        vif_arr.append(vif)
    
    # 출력
    for i in range(len(vif_name)):
        print(str(vif_name[i])+ " - " + str(vif_arr[i]))
    
    return vif_name, vif_arr

vif_crime = crime.drop(['Date'], axis = 1)
name_list = list(vif_crime)
vif(name_list, vif_crime)
