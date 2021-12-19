
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

# 모델생성및 평가


crime = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/final_merge2_crime_last.csv')

list(crime)
count = 0
for i in range(len(crime)):
    if crime['SHOOTING_Y'][i] == 1:
        continue
    if count == 1:
        count = 0
        print(count)
        continue
    
    if crime['UCR_PART_Part One'][i] == 1:
        crime['SHOOTING_Y'][i] = 1
        count += 1
        
count = 0
for i in range(len(crime)):
    count+=1 
    if crime['final_count'][i]>=1:
        crime['final_count'][i] = 1
    print(count)
     
crime.to_csv('C:/Users/dldyd/OneDrive/Desktop/happyBA.csv')   
# 평가
        
crime = crime.drop(['UCR_PART_Part One'],axis = 1)

count = 0
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


