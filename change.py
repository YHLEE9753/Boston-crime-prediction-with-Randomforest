import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon





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
"West Roxbury",]

crime = pd.read_csv('C:/Users/dldyd/crime3withLocation.csv')
crime['District'] = 0

base_dir = "data/"
All_poly = []
for name in boston_list:
    coords = []
    data_dir = base_dir + name + ".txt"
    f = open(data_dir, "r")
    lines = f.readlines()
    for line in lines:
        l = line.split(",")
        point = (float(l[0]), float(l[1]))
        coords.append(point)
    
    poly = Polygon(coords)
    All_poly.append(poly)
    f.close


# change 
count = 0
result = []
# for i in range(len(crime)):
for i in range(len(crime)):
    count+=1
    x = crime['Lat'][i]
    y = crime['Long'][i]
    point = Point(x,y)
    flag = False
    for i in range(len(All_poly)):
        if point.within(All_poly[i]):
            # crime['District'][i] = boston_list[i]
            result.append(boston_list[i])
            flag = True
            break
    if not flag:
        result.append('0')
        
print(len(crime))
print(len(result))
crime['District2'] = result

crime.to_csv('District3.csv')


