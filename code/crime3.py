import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


crime = pd.read_csv('C:/Users/dldyd/crime_final.csv')
list(crime)

###### 시간 오래 걸리는 부분
# final_count 최종 정리
crime['final_count'] = 0
date_list = list(set(list(crime['Date'])))
list(crime)
count = 0
district = list(set(list(crime['District2'])))
for i in district:
    for date in date_list:
        is_date = crime['Date'] == date
        is_problem = crime['UCR_PART_Part One']
        is_center = crime['District2'] == i
        index_list = crime[is_date & is_problem & is_center].index
        number = len(index_list)
        for n in index_list:
            crime['final_count'][n] = number
            
        count+=1
        print(count)
        
        
drop_list = ["STREET",'Lat','Long','HOUR','Date','REPORTING_AREA','DISTRICT','OFFENSE_CODE','Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1']
crime = crime.drop(drop_list,axis = 1)

vif_crime = crime.drop(['District2','final_count'], axis = 1)

############### vif 구하는 function
# vif 구하기
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

name_list = list(vif_crime)
vif(name_list, vif_crime)

crime = crime.drop(['school_count'],axis = 1)

crime.to_csv("crime_final2.csv")
