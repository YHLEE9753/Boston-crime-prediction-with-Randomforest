# final round

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
from sklearn import tree
from sklearn.metrics import accuracy_score
import datetime

# 모델생성및 평가
crime = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BA/save/final_merge2_crime.csv')
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

#Year, Month
crime.dtypes
crime['Date']= pd.to_datetime(crime['Date']) 
crime['Year'] = crime['Date'].dt.year 
crime['Month'] = crime['Date'].dt.month 


crime = crime.drop(del_list,axis = 1)
crime = crime.drop(['Unnamed: 0'], axis = 1)
crime = crime.drop(['P0020001'], axis = 1)
crime['Date'] = pd.to_datetime(crime['Date'])
crime_model = crime[:]
catvar=['GEOID']
#for c in catvar:
#    dummy = pd.get_dummies(crime_model[c], prefix=c, drop_first=True)
#    crime_model = pd.concat((crime_model,dummy),axis=1)

#for i in range(len(crime_model)):
#    if crime_model['final_count'][i]>=1:
#        crime_model['final_count'][i] = 1
    
    
X = crime_model.drop(catvar+['final_count','Date'],axis=1)
X.columns

list(crime)

y=crime_model['final_count']
X=sm.add_constant(X)


trainX, testX, trainY, testY = train_test_split(X,y, 
                                                    test_size=0.3, 
                                                    shuffle=True,
                                                    random_state=1004)


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
clf.fit(trainX, trainY)
y_pred = clf.predict(testX)
print('Decision Tree Accuracy: %.2f' % accuracy_score(testY, y_pred_tr))

# plot
#fig=plt.figure(dpi=900) 
#tree.plot_tree(clf)

# 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# instantiate the classifier 
rfc = RandomForestClassifier(random_state=0)
rfc.fit(trainX, trainY)
y_pred = rfc.predict(testX)
print('Model accuracy score with 10 decision-trees : {0:0.4f}'. 
      format(accuracy_score(testY, y_pred)))

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print(accuracy_score(testY, y_pred))	# 0.3
print(recall_score(testY, y_pred, labels=[1,2], average='micro'))	# 0.42
print(precision_score(testY, y_pred, labels=[1,2], average='micro'))	# 0.5
print(f1_score(testY, y_pred, labels=[1,2], average='micro'))	# 0.46

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
cm = confusion_matrix(testY, y_pred)



plt.rcParams['figure.figsize'] = [10, 8]
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix', fontsize=20)

plt.show()


