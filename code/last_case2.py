import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

census = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/data/census.csv')
crime = pd.read_csv('final_merge1_crime.csv')
list(crime)


import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon




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
crime = pd.read_csv('GEOID2.csv')
        

#######
list(crime)
del_list = ['Unnamed: 0',
 'Unnamed: 0.1', 'HOUR','Lat',
 'Long',
 'Location',
 'Year',
 'Month',
 'Day', 'Low Temp (F)',
 'High Humidity (%)',
 'Low Humidity (%)',
 'High Sea Level Press (in)',
 'Avg Sea Level Press (in)',
 'Low Sea Level Press (in)','High Wind (mph)',
 'Avg Wind (mph)',
 'High Wind Gust (mph)',
 'Snowfall (in)',
 'Precip (in)',
 'Events']
crime = crime.drop(del_list, axis = 1)

crime = pd.merge(crime,census, how = 'left', on = 'GEOID')

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
crime = crime.drop(['location'],axis = 1)
crime = crime.drop(['Unnamed: 0.1.1'],axis = 1)

# target
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

# 저장
crime.to_csv('GEOID3.csv')
crime = pd.read_csv('GEOID3.csv')
        
crime = crime.drop(['UCR_PART_Part One'],axis = 1)

count = 0
for i in range(len(crime)):
    count+=1 
    if crime['final_count'][i]>=1:
        crime['final_count'][i] = 1
    print(count)
crime = crime.drop(['Unnamed: 0', 'Date'], axis = 1)

cat=['DAY_OF_WEEK']
# 시간 더미 만들기
for c in cat:
    dummy = pd.get_dummies(crime[c], prefix=c, drop_first=False)
    crime = pd.concat((crime,dummy),axis=1)
crime = crime.drop(cat, axis = 1)

list(crime)
# correlation
data = crime.corr()
data = data.sort_values(by=['final_count'],axis=1, ascending=False)
data = data.iloc[range(16),0]
data = data.sort_values(ascending = False)
data = data.iloc[0:16]
print(data)




#####################################################
## 2016, 2017이랑 2018  쪼개기
crime_model = crime[:]
trainX = crime_model[crime_model['YEAR']<=2017]
trainY = trainX['final_count']
trainX = trainX.drop(['final_count'],axis=1)


testX=crime_model[crime_model['YEAR']>2017]
testY = testX['final_count']
testX = testX.drop(['final_count'],axis=1)



from sklearn import tree
from sklearn.metrics import accuracy_score


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)
clf.fit(trainX, trainY)
y_pred_tr = clf.predict(testX)
print('Decision Tree Accuracy: %.2f' % accuracy_score(testY, y_pred_tr))


# randomforest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(trainX, trainY)
y_pred = rfc.predict(testX)
print('RandomForest score : {0:0.4f}'. 
      format(accuracy_score(testY, y_pred)))

# evaluation
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('accuracy score: {:.4f}'.format(accuracy_score(testY, y_pred)))	# 0.3
print('recall score: {:.4f}'.format(recall_score(testY, y_pred)))	# 0.42
print('precision score: {:.4f}'.format(precision_score(testY, y_pred)))	# 0.5
print('f1 score: {:.4f}'.format(f1_score(testY, y_pred)))	# 0.46

# Confusion matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
cm = confusion_matrix(testY, y_pred)

plt.rcParams['figure.figsize'] = [10, 8]
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix', fontsize=20)

plt.show()


# feauture 중요도선정
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rfc.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = trainX.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Top 20 Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()






