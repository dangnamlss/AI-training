import rasterio
import math
import pandas as pd

path = "hcm1.tif"
path1 = "ndvi.tif"
path2 = "ndwi.tif"
ds = rasterio.open(path)
ds1 = rasterio.open(path1)
ds2 = rasterio.open(path2)

ndviData = ds1.read()
ndwiData = ds2.read()
data = ds.read()
data1 = ds.read(1)  #doc band 1
data2 = ds.read(2)  #doc band 2
data3 = ds.read(3)  #doc band 3
data4 = ds.read(4)  #doc band 4
data5 = ds.read(5)  #doc band 5
data6 = ds.read(6)  #doc band 6
data7 = ds.read(7)  #doc band 7

csvFile = pd.read_csv(r'data.csv')

boundImage = ds.bounds
imageHeight = ds.height
imageWidth = ds.width

leftImage = boundImage[0]
topImage = boundImage[3]
pixelSizeX = ds.transform[0]
pixelSizeY = -ds.transform[4]

def calculateIndex(x,y):
    row = (topImage - y)/pixelSizeY
    col = (x - leftImage)/pixelSizeX
    return math.floor(row),math.floor(col)
print(leftImage)
print(pixelSizeX)
print(calculateIndex(677283.3, 1187989))


def getBandsColumn(bandMatrix):
    res = []
    for i in range(0,csvFile['X'].count(),1):
        x = csvFile['X'][i]
        y = csvFile['Y'][i]
        row, col = calculateIndex(x,y)
        if((row <= imageHeight and row > 0) and (col <= imageWidth and col > 0)):
            res.append(bandMatrix[row][col])
        else:
            print("hello")
            res.append(None)
    return res


band1 = getBandsColumn(data1)
csvFile['Band_1'] = band1
band2 = getBandsColumn(data2)
csvFile['Band_2'] = band2
band3 = getBandsColumn(data3)
csvFile['Band_3'] = band3
band4 = getBandsColumn(data4)
csvFile['Band_4'] = band4
band5 = getBandsColumn(data5)
csvFile['Band_5'] = band5
band6 = getBandsColumn(data6)
csvFile['Band_6'] = band6
band7 = getBandsColumn(data7)
csvFile['Band_7'] = band7
ndviFile = getBandsColumn(ndviData[0])
csvFile['NDVI'] = ndviFile
ndwiFile = getBandsColumn(ndwiData[0])
csvFile['NDWI'] = ndwiFile
csvFile.to_csv('final_1.csv',index=False)