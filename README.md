# Boston-crime-prediction-with-Randomforest
Boston crime prediction with Decision Tree, Randomforest, KNN

# Description
We predict boston crime using machine learning.<br>
hypothesis is that since there is some racism in the United States, there will be many violent crimes against people of color.<br> 
Also according to EDA, Crime rates occur a lot in the summer season when the temperature is high and humidity is high.<br>

# Data source
Crime data (crime data in Boston) from Kaggle<br>
Shooting data (shooting crime data in Boston) from Official Boston Dataset site<br>
Weather data (weather data in Boston) from Kaggle<br>
BFD data (firearm recovery count in Boston) from Kaggle<br>
Census data (USA District population data) using QGIS program with official census site<br>

# Method
## learning methond
1. Decision Tree
2. RandomForest
3. KNN

## evaluate method
1. accuracy
2. cv_score
3. confusion matrix
4. precision
5. recall
6. f1 score
7. support

## additional method to upgrade
1. Oversampling = SMOTE, ADASYN
2. ensemble method = AdaBoost, GradientBoost

# Evaluation
Decsion Tree accuracy = 0.9298<br>
RandomForest accuracy = 0.9269<br>
KNN accuracy = 0.7959<br>
<br>
Accuracy of SMOTE in RandomForest = 0.7190<br>
Accuracy of ADASYN in RandomForest = 0.6742<br>
<br>
Accuracy of AdaBoost in RandomForest = 0.9298<br>
Accuracy of GradientBoost in RandomForest = 0.9298<br>
<br>
Best ROC_AUC score is using AdaBoosting = 0.8658<br>

# Conclusion
1. Our model can predict the likelihood of crime with about 92% accuracy. However, our model is judged to have poor predictive power for crime occurrence(final_count=1).<br>
2. Violent crimes have a high connection to gun accidents, and crimes using firearms are likely to occur as violent crimes.<br>
3. The occurrence of violent crimes was less affected by other variables such as weather and gun accident victims.<br>

# more detail
You can see detail in ppt file



