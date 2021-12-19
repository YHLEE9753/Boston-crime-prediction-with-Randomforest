import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# 모델생성및 평가
crime = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data2/final_merge2_crime.csv')
list(crime)
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
crime = crime.drop(['Unnamed: 0'], axis = 1)
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

vif_crime = crime.drop(['Date', 'final_count'], axis = 1)
name_list = list(vif_crime)
vif(name_list, vif_crime)

list(vif_crime)
vif_crime = vif_crime.drop(['P0020001'], axis = 1)


crime = crime.drop(['P0020001'], axis = 1)



crime['Date'] = pd.to_datetime(crime['Date'])
crime_model = crime[:]
catvar=['GEOID']
for c in catvar:
    dummy = pd.get_dummies(crime_model[c], prefix=c, drop_first=True)
    crime_model = pd.concat((crime_model,dummy),axis=1)
    
    
X = crime_model.drop(catvar+['final_count','Date'],axis=1)
X.columns


y=crime_model['final_count']
X=sm.add_constant(X)

model=sm.OLS(y,X.astype(float))
result = model.fit()
result.summary()

trainX, testX, trainY, testY = train_test_split(X,y, 
                                                    test_size=0.2, 
                                                    shuffle=True,
                                                    random_state=1004)


print('trainX shape:', trainX.shape)
print('testX shape:', testX.shape)
print('trainY shape:', trainY.shape)
print('testY shape:', testY.shape)



# Model
reg1=LinearRegression()
reg2=Ridge(alpha=1)
reg3=Lasso(alpha=1)

#fit
reg1.fit(trainX,trainY)
reg2.fit(trainX,trainY)
reg3.fit(trainX,trainY)

#Linear Regression
print('Linear Regression(train):',reg1.score(trainX,trainY))
print('Linear Regression(test):',reg1.score(testX,testY))

#Ridge Regression
print('Ridge Regression(train):',reg2.score(trainX,trainY))
print('Ridge Regression(test):',reg2.score(testX,testY))


#Lasso Regression
print('Lasso Regression(train):',reg3.score(trainX,trainY))
print('Lasso Regression(test):',reg3.score(testX,testY))
