import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

crime = pd.read_csv('C:/Users/dldyd/OneDrive/Desktop/BACrime/District3.csv')
school = pd.read_csv(
    'C:/Users/dldyd/OneDrive/Desktop/BA_data/Location/Public_Schools.csv')
police = pd.read_csv(
    'C:/Users/dldyd/OneDrive/Desktop/BA_data/Location/Boston_Police_Stations.csv')
landmark = pd.read_csv(
    'C:/Users/dldyd/OneDrive/Desktop/BA_data/Location/Boston_Landmarks_Commission_(BLC)_Landmarks.csv')
crime = crime.drop(['District'], axis=1)

crime.dtypes

idx = crime[crime['District2'] == '0'].index
crime = crime.drop(idx)

crime.size

boston_list = [
    "Allston",
    "Back Bay",
    "Bay Village",
    "Beacon Hill",
    "Brighton",
    "Charlestown",
    "Chinatown",
    "Dorchester",
    "Downtown",
    "East Boston",
    "FenWay-Kenmore",
    "Hyde Park",
    "Jamaica Plain",
    "Leather District",
    "Mattapan",
    "Mission Hill",
    "North End",
    "Rosilndale",
    "Roxbury",
    "South End",
    "South Boston",
    "West End",
    "West Roxbury", ]

for i in range(len(school)):
    if school['CITY'][i] == 'Roslindale':
        school['CITY'][i] = 'Rosilndale'
    if school['CITY'][i] == 'Boston':
        school['CITY'][i] = '0'
idx2 = school[school['CITY'] == '0'].index
school = school.drop(idx2)
school_drop = list(school)
school_drop.remove('CITY')
school = school.drop(school_drop, axis=1)
school['count'] = 1
school['final'] = 0

school = school.reset_index(drop=True)
result = []
for i in range(len(school)):
    string = school['CITY'][i]
    idx = school[school['CITY'] == string].index
    size = len(idx)
    result.append(size)
result
school['final'] = result
school = school.drop(['count'], axis=1)

####################################################


for i in range(len(police)):
    if police['NEIGHBORHOOD'][i] == 'Boston':
        police['NEIGHBORHOOD'][i] == '0'

police = police.reset_index(drop=True)
idx3 = police[police['NEIGHBORHOOD'] == '0'].index
police = police.drop(idx3)
police_drop = list(police)
police_drop.remove('NEIGHBORHOOD')
police = police.drop(police_drop, axis=1)
police['count'] = 1

for i in range(len(police)):
    if police['NEIGHBORHOOD'][i] == 'Boston':
        police['NEIGHBORHOOD'][i] == '0'

result2 = []
for i in range(len(police)):
    idx = police[police['NEIGHBORHOOD'] == police['NEIGHBORHOOD'][i]].index
    size = len(idx)
    result2.append(size)
result2
police['final'] = result2
police = police.drop(['count'], axis=1)

####################################################
for i in range(len(landmark)):
    if landmark['Neighborho'][i] == 'Fenway':
        landmark['Neighborho'][i] = 'FenWay-Kenmore'
    if landmark['Neighborho'][i] == 'Fenway, JP':
        landmark['Neighborho'][i] = 'FenWay-Kenmore'
    if landmark['Neighborho'][i] == 'Waterfront':
        landmark['Neighborho'][i] = 'South Boston'

    if landmark['Neighborho'][i] == 'Theater':
        landmark['Neighborho'][i] = '0'
    if landmark['Neighborho'][i] == 'Beacon Hill/Back Bay':
        landmark['Neighborho'][i] = '0'
    if landmark['Neighborho'][i] == 'South Cove':
        landmark['Neighborho'][i] = '0'
    if landmark['Neighborho'][i] == 'Theater District':
        landmark['Neighborho'][i] = '0'
    if landmark['Neighborho'][i] == 'Boston':
        landmark['Neighborho'][i] = '0'
    if landmark['Neighborho'][i] == ' ':
        landmark['Neighborho'][i] = '0'
idx2 = landmark[landmark['Neighborho'] == '0'].index
landmark = landmark.drop(idx2)

landmark_drop = list(landmark)
landmark_drop.remove('Neighborho')
landmark = landmark.drop(landmark_drop, axis=1)
landmark['count'] = 1
landmark['final'] = 0

landmark = landmark.reset_index(drop=True)
result3 = []
for i in range(len(landmark)):
    string = landmark['Neighborho'][i]
    idx = landmark[landmark['Neighborho'] == string].index
    size = len(idx)
    result3.append(size)
result3
landmark['final'] = result3
landmark = landmark.drop(['count'], axis=1)


print(set(list(school['CITY'])))
print(set(list(police['NEIGHBORHOOD'])))
print(set(list(landmark['Neighborho'])))


school.columns=['District2','school_count']
police.columns=['District2','police_count']
landmark.columns=['District2','landmark_count']

school = school.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
police = police.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
landmark = landmark.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

crime=crime.merge(school, on='District2',how = 'inner')
crime=crime.merge(police, on='District2',how = 'inner')
crime=crime.merge(landmark, on='District2',how = 'inner')

crime.to_csv("crime_final.csv")


