import pandas as pd
import rasterio
import numpy

from sklearn.metrics import classification_report

data_path = r'usable data.csv'
img_path = r'hcm1.tif'
ds = rasterio.open(img_path)
df = pd.read_csv(data_path)

X_labels = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7']
y_label = 'Label_ID'

### ----------------------------------------------------------------------------------
## CHIA DỮ LIỆU THÀNH TẬP TRAIN - TEST THEO TỈ LỆ 2 - 8
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[X_labels], df[y_label], test_size=0.2, stratify=df['Label_ID'], random_state=15)

### ------------------------------------------------------------------------------------
def validModel (Model, X_test, y_test):
    y_pred = Model.predict(X_test)
    print(classification_report(y_test, y_pred))

### -----------------------------------------------------------------------------------
## SỬ DỤNG MODAL ĐÁNH GIÁ DỮ LIỆU
## Gradient Boost
# Chia tập huấn luyện thành 10 fold
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

clf1 = GradientBoostingClassifier(n_estimators=10, random_state=9)

skf1 = StratifiedKFold(n_splits=10, shuffle=True,random_state=12)

cv_scores = cross_val_score(clf1, X_train, y_train, cv=skf1, scoring='accuracy')

print('Accuracy score - 10foldCV - Training dataset: {}'.format(np.mean(cv_scores)))

# Đánh giá tập test
clf1.fit(X_train,y_train)
y_pred=clf1.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))

# Tập train
clf1.fit(X_train,y_train)
y_pred=clf1.predict(X_train)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy score - Train dataset: {}'.format(accuracy_score(y_train, y_pred)))

# Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 150],
    'max_features': ['auto', 'sqrt','log2']
}

gridCV_clf = GridSearchCV(estimator = clf1, param_grid=param_grid, cv = skf1, scoring='accuracy', verbose=2)
gridCV_clf.fit(X_train, y_train)
print(gridCV_clf.best_params_)
print(gridCV_clf.best_score_)

# Sử dụng tham số mới sau khi tối ưu:
optimizedClf = GradientBoostingClassifier(max_features='sqrt', n_estimators=150)
optimizedClf.fit(X_train,y_train)
y_pred=optimizedClf.predict(X_test)

validModel(optimizedClf, X_test, y_test)

## Random Forest
# Chia tập huấn luyện thành 10 fold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

clf2=RandomForestClassifier(n_estimators=100, random_state=9)

skf2 = StratifiedKFold(n_splits=10, shuffle=True,random_state=12)

cv_scores = cross_val_score(clf2, X_train, y_train, cv=skf2, scoring='accuracy')

print('Accuracy score - 10foldCV - Training dataset: {}'.format(np.mean(cv_scores)))

# Đánh giá tập test
clf2.fit(X_train,y_train)
y_pred=clf2.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))

# Tập train
clf2.fit(X_train,y_train)
y_pred=clf2.predict(X_train)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy score - Train dataset: {}'.format(accuracy_score(y_train, y_pred)))

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [10, 150],
    'max_features': ['auto', 'sqrt','log2'],
    'criterion' : ['gini','entropy']
}

gridCV_clf2 = GridSearchCV(estimator = clf2, param_grid=param_grid, cv = skf2, scoring='accuracy', verbose=2)
gridCV_clf2.fit(X_train, y_train)
print(gridCV_clf2.best_params_)
print(gridCV_clf2.best_score_)

# Sử dụng tham số mới sau khi tối ưu:
optimizedClf2 = RandomForestClassifier(max_features='auto', n_estimators=150, criterion='entropy')
optimizedClf2.fit(X_train,y_train)
y_pred=optimizedClf2.predict(X_test)

# from sklearn.metrics import accuracy_score, confusion_matrix
# print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))

validModel(optimizedClf2, X_test, y_test)

## Get pixels
model = optimizedClf2
finalData = []
band1 = ds.read(1)
band2 = ds.read(2)
band3 = ds.read(3)
band4 = ds.read(4)
band5 = ds.read(5)
band6 = ds.read(6)
band7 = ds.read(7)

for i in range (0, band1.shape[0], 1):
    array = []
    for j in range(0,band1.shape[1],1):
        if(band1[i][j] != band1[i][j]):
            array.append(None)
            continue

        x = [[band1[i][j],band2[i][j],band3[i][j],band4[i][j],band5[i][j],band6[i][j],band7[i][j]]]
        y = optimizedClf2.predict(x)
        array.append(y[0])
    finalData.append(array)

## Gen map
from rasterio.crs import CRS
from rasterio.transform import from_origin
crs = CRS.from_epsg(3424)
transform = from_origin(648129.923572, 721997.12265, 29.8092006, 29.8092006) #tham số: left, right, x, y

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
from rasterio.transform import from_origin
crs = CRS.from_epsg(3424)
transform = from_origin(648129.923572, 721997.12265, 29.8092006, 29.8092006) #tham số: left, right, x, y

with rasterio.open("finalImage.tif", "w",
                   driver= driver,
                   height= height,
                   width= width,
                   count= count,
                   dtype= dtype,
                   crs= crs,
                   transform= transform) as dst:
        dst.write(finalImageTiff)
print("done")