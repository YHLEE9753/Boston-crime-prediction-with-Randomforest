# final 이전 최종 테스트1

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

crime = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/crime.csv')
weather = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/Boston weather_clean_1.csv')
BPD = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/BPD FIREARM RECOVERY COUNTS_1.csv')
shooting = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/Shooting_1.csv')
census = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/census.csv')

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
#######################
############### crime data perprocessing
crime.shape
crime.dtypes
print("===== Boston Crime =====")
for name in crime:
    print("NaN value of " + name + " : " + str(crime[name].isnull().sum()))
    
# 위도 경도 Nan row 전부 제거
crime = crime.dropna(subset=['Lat','Long'])
# UCR part 가 없으면 범죄 분류가 안되므로 Nan 값 제거
crime = crime.dropna(subset=['UCR_PART'])
# shooting, district, street nan 값 0 으로 할당
crime = crime.replace(np.NaN,0)
# 위도 경도로 처리할것이기 때문에 district, street 값 활용하지 않는다.
# location 은 사용하지 않는다.
# crime = crime.drop(['STREET','DISTRICT','Location'], axis = 1)
crime = crime.drop(['INCIDENT_NUMBER','OFFENSE_CODE','OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION','DISTRICT','REPORTING_AREA','STREET'], axis = 1)
#crime = crime.drop(['YEAR','MONTH','DAY_OF_WEEK','HOUR'], axis = 1)


# shooting 더미컬럼으로 처리하기
crime = pd.get_dummies(crime, columns = ['SHOOTING'], drop_first=True)

# 1차적 타켓 생성(UCR part 1)
# violent columns
crime = pd.get_dummies(crime, columns = ['UCR_PART'], drop_first=True)
crime = crime.drop(['UCR_PART_Part Three', 'UCR_PART_Part Two'], axis = 1)

# index 초기화
crime = crime.reset_index(drop=True)


############### 날짜 변경 for merge
for i in range(len(crime)):
    a = crime['OCCURRED_ON_DATE'][i][:4]
    b = crime['OCCURRED_ON_DATE'][i][5:7]
    c = crime['OCCURRED_ON_DATE'][i][8:10]
    date = a+b+c
    crime['OCCURRED_ON_DATE'][i] = date
    print(i)

####################################### save point 1
# csv 저장(저장하기)
#crime.to_csv('final_changedate_crime.csv')
crime = pd.read_csv('final_changedate_crime.csv')
crime2 = crime[:]
crime = crime2[:]
############### weather 과 BPD 와 shooting 과 merge 하자
# weather 과 merge
BPD.rename(columns={'CollectionDate':'Date'},inplace = True)
crime.rename(columns={'OCCURRED_ON_DATE':'Date'},inplace = True)
shooting = shooting.drop(['Date'], axis = 1)
shooting.rename(columns={'Shooting_Date':'Date'},inplace = True)
BPD['Date'] = pd.to_datetime(BPD['Date'])
weather['Date'] = pd.to_datetime(weather['Date'])
crime['Date'] = pd.to_datetime(crime['Date'])
shooting['Date'] = pd.to_datetime(shooting['Date'])
crime.rename(columns = {'OCCURRED_ON_DATE' : 'Date'}, inplace = True)


weather['Date2'] = 0
for i in range(len(weather)):
    a = str(weather['Date'][i])[:4]
    b = str(weather['Date'][i])[5:7]
    c = str(weather['Date'][i])[8:10]
    date = a+b+c
    weather['Date2'][i] = date
    print(date)
weather = weather.drop(['Date'], axis = 1)
weather.rename(columns = {'Date2' : 'Date'}, inplace = True)

# 날짜 조정
# 2015-6-15 ~ 2018-9-3
mask1 = (shooting['Date'] >= '2015-06-15') & (shooting['Date'] <= '2018-09-03')
shooting=shooting.loc[mask1]

mask2 = (BPD['Date'] >= '2015-06-15') & (BPD['Date'] <= '2018-09-03')
BPD=BPD.loc[mask2]
BPD
crime = crime.merge(weather, how = 'inner', on='Date')
#crime = pd.merge(crime,BPD, how = 'left', on = 'Date')
# = pd.merge(crime,shooting, how = 'left', on = 'Date')

crime = crime.merge(BPD, how = 'inner', on='Date')
crime = crime.merge(shooting, how = 'inner', on='Date')


# 불필요한 column 제거
print(list(crime))
crime = crime.drop(['High Sea Level Press (in)', 'Avg Sea Level Press (in)', 'Low Sea Level Press (in)'], axis = 1)


# 마지막으로 data preprocessing 하자
crime.shape
crime.dtypes
print("===== Boston Crime =====")
for name in crime:
    print("NaN value of " + name + " : " + str(crime[name].isnull().sum()))
crime = crime.replace(np.NaN,0)
    

# vif 를 구해보자
vif_crime = crime.drop(['REPORTING_AREA','Events','Shooting_Type_V2','Victim_Gender','Victim_Race','Victim_Ethnicity_NIBRS','Multi_Victim'], axis = 1)
vif_crime = vif_crime.drop(['Date'], axis = 1)
name_list = list(vif_crime)
vif(name_list, vif_crime)

# vif 큰거를 제거하자
vif_crime = vif_crime.drop(['High Visibility (mi)'], axis = 1)
vif_crime = vif_crime.drop(['Avg Temp (F)'], axis = 1)
vif_crime = vif_crime.drop(['Avg Humidity (%)'], axis = 1)
vif_crime = vif_crime.drop(['Avg Dew Point (F)'], axis = 1)
vif_crime = vif_crime.drop(['Low Dew Point (F)'], axis = 1)
vif_crime = vif_crime.drop(['High Dew Point (F)'], axis = 1)
vif_crime = vif_crime.drop(['High Temp (F)'], axis = 1)
vif_climate = ['High Visibility (mi)','Avg Temp (F)','Avg Humidity (%)','Avg Dew Point (F)','Low Dew Point (F)','High Dew Point (F)','High Temp (F)']
# name_list = list(vif_crime)
# vif(name_list, vif_crime)

crime = crime.drop(vif_climate, axis = 1)
crime.dtypes

# gun data 를 더미로 만든 후 vif 를 구해보자
vif_crime2 = crime.drop(['REPORTING_AREA', 'Events','Date'], axis = 1)

vif_crime2 = pd.get_dummies(vif_crime2, columns = ['Shooting_Type_V2'], drop_first=True)
vif_crime2 = pd.get_dummies(vif_crime2, columns = ['Victim_Gender'], drop_first=True)
vif_crime2 = pd.get_dummies(vif_crime2, columns = ['Victim_Race'], drop_first=True)
vif_crime2 = pd.get_dummies(vif_crime2, columns = ['Victim_Ethnicity_NIBRS'], drop_first=True)
vif_crime2 = pd.get_dummies(vif_crime2, columns = ['Multi_Victim'], drop_first=True)

vif_crime2 = vif_crime2.drop(['Victim_Ethnicity_NIBRS_Unknown'], axis = 1)
vif_crime2 = vif_crime2.drop(['Victim_Race_Unknown'], axis = 1)
vif_crime2 = vif_crime2.drop(['High Wind (mph)'], axis = 1)
vif_gun = ['Victim_Ethnicity_NIBRS_Unknown','Victim_Race_Unknown','High Wind (mph)']
# name_list = list(vif_crime2)
# vif(name_list, vif_crime2)

# 최종 부분
crime = pd.get_dummies(crime, columns = ['Shooting_Type_V2'], drop_first=True)
crime = pd.get_dummies(crime, columns = ['Victim_Gender'], drop_first=True)
crime = pd.get_dummies(crime, columns = ['Victim_Race'], drop_first=True)
crime = pd.get_dummies(crime, columns = ['Victim_Ethnicity_NIBRS'], drop_first=True)
crime = pd.get_dummies(crime, columns = ['Multi_Victim'], drop_first=True)
crime = crime.drop(vif_gun, axis = 1)
crime = crime.drop(['Events'], axis = 1)

# csv 저장(저장하기)
crime = crime.drop(['Incident_Num','District'], axis = 1)
crime.to_csv('final_merge1_crime.csv')



###### cencus merge 하기
#이거는 vs code 에서 하자



###### 시간 오래 걸리는 부분
# final_count 최종 정리
#crime['final_count'] = 0
#date_list = list(set(list(crime['Date'])))

#count = 0
#for i in range(500):
#    for date in date_list:
#        is_date = crime['Date'] == date
#        is_problem = crime['UCR_PART_Part One']
#        is_center = crime['cluster_no'] == i
#        index_list = crime[is_date & is_problem & is_center].index
#        number = len(index_list)
#        for n in index_list:
#            crime['final_count'][n] = number
            
#        count+=1
#        print(count)
        
#############


# csv 저장(저장하기)
crime.to_csv('crime4.csv')



    