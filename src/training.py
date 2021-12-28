import numpy as np
import pandas as pd
import rasterio
import numpy
from numpy import asarray
import os

from sklearn.metrics import classification_report

data_path = r'usable data.csv'
img_path = r'hcm1.tif'
ds = rasterio.open(img_path)
df = pd.read_csv(data_path)

X_labels = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7']
y_label = 'Label_ID'

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[X_labels], df[y_label], test_size=0.2, stratify=df['Label_ID'], random_state=15)

def validModel (Model, X_test, y_test):
    y_pred = Model.predict(X_test)
    print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------------------------
# SỬ DỤNG MODAL ĐÁNH GIÁ DỮ LIỆU
# Gradient Boost

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(max_features=0.2, n_estimators=150)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

validModel(clf, X_test, y_test)

#-------------------------------------------------------------------------------------
# TRAINING
model = clf
finalData = []
band1 = ds.read(1)
band2 = ds.read(2)
band3 = ds.read(3)
band4 = ds.read(4)
band5 = ds.read(5)
band6 = ds.read(6)
band7 = ds.read(7)


t = 0
for i in range (0, band1.shape[0], 1):
    print(t)
    t = t + 1
    array = []
    for j in range(0,band1.shape[1],1):
        if(band1[i][j] != band1[i][j]):
            array.append(None)
            continue

        x = [[band1[i][j],band2[i][j],band3[i][j],band4[i][j],band5[i][j],band6[i][j],band7[i][j]]]
        y = model.predict(x)
        array.append(y[0])
    finalData.append(array)

finalImageArr = numpy.array(finalData)
finalImage = []
finalImage.append(finalImageArr)

finalImageTiff = numpy.array(finalImage) #numpy array gồm 3 dimensions

dim1 = finalImageTiff.shape
driver = "GTiff"
height = dim1[1]
width = dim1[2]
count = 1
dtype = finalImageTiff.dtype
dtype = np.float32
from rasterio.crs import CRS
crs = CRS.from_epsg(3424)
from rasterio.transform import from_origin
transform = from_origin(648129.923572, 721997.12265, 29.8092006, 29.8092006) #tham số: left, right, x, y

with rasterio.open("finalImage.tif",
                    "w", 
                    driver = driver, 
                    height= height, 
                    width = width, 
                    count = count, 
                    dtype = dtype,
                    crs = crs, transform = transform) as dst:
    dst.write(finalImageTiff)
print("done")





