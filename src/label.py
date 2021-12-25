import csv
import rasterio
import math
import pandas as pd

# INDEXING LABEL - ĐÁNH NHÃN CHO DỮ LIỆU
raw_data = pd.read_csv(r'final_1.csv')

def indexLabelCol ():
    res = []
    for i in range(0, raw_data['ID'].count(), 1):
        if(raw_data['Label'][i] == 'Residental'):
            res.append(1)
        elif (raw_data['Label'][i] == 'Rice paddies'):
            res.append(2)
        elif (raw_data['Label'][i] == 'Cropland'):
            res.append(3)
        elif (raw_data['Label'][i] == 'Grass'):
            res.append(4)
        elif (raw_data['Label'][i] == 'Barren land'):
            res.append(5)
        elif (raw_data['Label'][i] == 'Forest'):
            res.append(6)
        elif (raw_data['Label'][i] == 'Wetland'):
            res.append(7)
        elif (raw_data['Label'][i] == 'Aquaculture'):
            res.append(8)
        elif (raw_data['Label'][i] == 'Water'):
            res.append(9)
        elif (raw_data['Label'][i] == 'Scrub'):
            res.append(10)
    return res

print(raw_data.head())
raw_data['Label_ID'] = indexLabelCol()
raw_data.to_csv('final raw data_1.csv', index=False)

#  XOÁ DỮ LIỆU LỖI
df = pd.read_csv(r'final raw data_1.csv')
df = df.dropna(how='any', axis=0)
df.to_csv('usable data.csv', index=False)
