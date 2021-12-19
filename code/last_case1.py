# final 이전 최종 테스트2

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

census = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/census.csv')

district_dict = {}

for i in range(len(census)):
    district_dict[census["GEOID"][i]] = census["location"][i]

del_list = []
for k in district_dict.keys():
    if type(k) == str:
        if len(k) != 11:
            del_list.append(k)
    else:
        del_list.append(k)
    
for d in del_list:
    del district_dict[d]
    
# 전처리 완료
# 리스트화
for k in district_dict.keys():
    a = district_dict[k]
    a = a[1:-1]
    a_list = a.split(',')
    number_list = []
    for i in range(len(a_list)):
        number = 0
        if i == 0:
            number = a_list[i][2:-1]
            number_list.append(number)
        elif i == len(a_list) - 1:
            number = a_list[i][2:-2]
            number_list.append(number)
        elif i%3 == 0:
            number = a_list[i][3:-1]
            number_list.append(number)
        elif i%3 == 1:
            number = a_list[i][2:]
            number_list.append(number)
        else:
            pass
    district_dict[k] = number_list


# 포인트화
for k in district_dict.keys():
    a = district_dict[k]
    for i in range(0,len(a),2):
        x = a[i]
        y = a[i+1]
        print(x)
        print(y)
        






