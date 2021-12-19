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
crime['Date']= pd.to_datetime(crime['Date']) 
crime['YEAR'] = crime['Date'].dt.year 
crime['MONTH'] = crime['Date'].dt.month 

# 시간 컨트롤
cat=['YEAR','MONTH']
# 시간 더미 만들기
for c in cat:
    dummy = pd.get_dummies(crime[c], prefix=c, drop_first=True)
    crime = pd.concat((crime,dummy),axis=1)
crime = crime.drop(cat, axis = 1)

crime = crime.drop(del_list,axis = 1)
crime = crime.drop(['Unnamed: 0'], axis = 1)
crime = crime.drop(['P0020001'], axis = 1)
crime = crime.drop(['Date'], axis = 1)
crime_model = crime[:]
catvar=['GEOID']
#for c in catvar:
#    dummy = pd.get_dummies(crime_model[c], prefix=c, drop_first=True)
#    crime_model = pd.concat((crime_model,dummy),axis=1)

#for i in range(len(crime_model)):
#    if crime_model['final_count'][i]>=1:
#        crime_model['final_count'][i] = 1
    
    


## 2016, 2017이랑 2018  쪼개기
is_train1 = crime_model['YEAR_2016'] == 1 
is_train2 = crime_model['YEAR_2017'] == 1
train_model = crime_model[is_train1 | is_train2]


is_test = crime_model['YEAR_2018'] == 1
test_model = crime_model[is_test]

trainX = train_model.drop(['final_count'], axis = 1)
testX = test_model.drop(['final_count'], axis = 1)
trainY = train_model['final_count']
testY = test_model['final_count']


# Decision Tree 로 평가
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
clf.fit(trainX, trainY)
y_pred = clf.predict(testX)
print('Decision Tree Accuracy: %.2f' % accuracy_score(testY, y_pred))


# Decision Tree plot 찍기
tree.plot_tree(clf)

# 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# instantiate the classifier 
rfc = RandomForestClassifier(random_state=0)
rfc.fit(trainX, trainY)
y_pred = rfc.predict(testX)
print('RandomForest score : {0:0.4f}'. 
      format(accuracy_score(testY, y_pred)))

# 하이퍼 파라미터 선정
from sklearn.model_selection import GridSearchCV

params = { 'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

# RandomForestClassifier ,GridSearchCV
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(trainX, trainY)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

#위의 결과로 나온 최적 하이퍼 파라미터로 다시 모델을 학습하여 테스트 세트 데이터에서 예측 성능을 측정
rf_clf1 = RandomForestClassifier(n_estimators = 100, 
                                max_depth = 12,
                                min_samples_leaf = 8,
                                min_samples_split = 20,
                                random_state = 0,
                                n_jobs = -1)




from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('accuracy score: {:.4f}'.format(accuracy_score(testY, y_pred)))	# 0.3
print('recall score: {:.4f}'.format(recall_score(testY, y_pred, labels=[1,2], average='micro')))	# 0.42
print('precision score: {:.4f}'.format(precision_score(testY, y_pred, labels=[1,2], average='micro')))	# 0.5
print('f1 score: {:.4f}'.format(f1_score(testY, y_pred, labels=[1,2], average='micro')))	# 0.46

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
cm = confusion_matrix(testY, y_pred)



plt.rcParams['figure.figsize'] = [10, 8]
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix', fontsize=20)

plt.show()


#roc
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    
from sklearn.metrics import roc_curve
prob = rfc.predict_proba(testX)
prob = probs[:, 1]
prob = rfc.predict_proba(testX)[:,1]
fper, tper, thresholds = roc_curve(testY, prob)
plot_roc_curve(fper, tper)

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

