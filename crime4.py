import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


crime = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data2/final_merge1_crime.csv')
census = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/census.csv')


# 리스트 전처리
district_dict = {}

for i in range(len(census)):
    district_dict[census["GEOID"][i]] = census["location"][i]

# del_list = []
# for k in district_dict.keys():
#     if type(k) == str:
#         if len(k) != 11:
#             del_list.append(k)
#     else:
#         del_list.append(k)
    
# for d in del_list:
#     del district_dict[d]
    
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
All_poly = []
for k in district_dict.keys():
    if k == 25025980101:
        continue
    a = district_dict[k]
    coords = []
    for i in range(0,len(a),2):
        x = a[i]
        y = a[i+1]
        print(k)

        point = (float(x),float(y))
        coords.append(point)
    coords.append((float(a[0]),float(a[1])))
    poly = Polygon(coords)
    All_poly.append(poly)

# change 
count = 0
result = []
boston_list = list(district_dict.keys())
# for i in range(len(crime)):
for i in range(len(crime)):
    count+=1
    y = crime['Lat'][i]
    x = crime['Long'][i]
    point = Point(x,y)
    flag = False
    for j in range(len(All_poly)):
        if point.within(All_poly[j]):
            # crime['District'][i] = boston_list[i]
            result.append(boston_list[j])
            flag = True
            print("!!")
            break
    if not flag:
        result.append(25025980101)
        # result.append('0')
        
print(len(crime))
print(len(result))
crime['GEOID'] = result

crime.to_csv('GEOID2.csv')
        








# district_dict = {}
# for i in census:


# All_poly = []
# for i in range(len(census)):


# for name in boston_list:
#     coords = []
#     data_dir = base_dir + name + ".txt"
#     f = open(data_dir, "r")
#     lines = f.readlines()
#     for line in lines:
#         l = line.split(",")
#         point = (float(l[0]), float(l[1]))
#         coords.append(point)
    
#     poly = Polygon(coords)
#     All_poly.append(poly)
#     f.close


# # change 
# count = 0
# result = []
# # for i in range(len(crime)):
# for i in range(len(crime)):
#     count+=1
#     x = crime['Lat'][i]
#     y = crime['Long'][i]
#     point = Point(x,y)
#     flag = False
#     for i in range(len(All_poly)):
#         if point.within(All_poly[i]):
#             # crime['District'][i] = boston_list[i]
#             result.append(boston_list[i])
#             flag = True
#             break
#     if not flag:
#         result.append('0')
        
# print(len(crime))
# print(len(result))
# crime['District2'] = result

# crime.to_csv('District3.csv')


