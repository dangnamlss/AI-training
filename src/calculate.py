from matplotlib import pyplot
#C:/Users/Nam/PycharmProjects/pythonProject1/hcm.tif

#MO FILE TIFF
import rasterio
path = "hcm.tif"
ds = rasterio.open(path)

data = ds.read()
data1 = ds.read(1)  #doc band 1
data2 = ds.read(2)  #doc band 2
data3 = ds.read(3)  #doc band 3
data4 = ds.read(4)  #doc band 4
data5 = ds.read(5)  #doc band 5
data6 = ds.read(6)  #doc band 6
data7 = ds.read(7)  #doc band 7

#HIEN THI ANH THEO BAND
# pyplot.imshow(data1, cmap='pink')
# pyplot.show()

#TINH MIN, MAX, MEAN, MEDIAN, STD CUA TUNG BAND
#Vi du Band 1
import numpy
print("Max:")
print(numpy.nanmax(data1))
print("Min:")
print(numpy.nanmin(data1))
print("Mean:")
print(numpy.nanmean(data1))
print("Median:")
print(numpy.nanmedian(data1))
print("Std:")
print(numpy.nanstd(data1))

#TINH CHI SO NDVI, NDWI
red = data4
green = data3
nir = data5 #band 5 - near infrared

ndviValueList = [] #ndvi = (nir - red) / (nir + red)
ndwiValueList = [] #ndwi = (nir - swir) / (nir + swir)

for i in range(0, red.shape[0], 1):
    arr1 = [] #store ndvi value
    arr2 = [] #store ndwi value
    for j in range(0, red.shape[1], 1):
        if (red[i][j] == None):
            arr1.append(None)
            arr2.append(None)
            continue
        value1 = (nir[i][j] - red[i][j]) / (nir[i][j] + red[i][j])
        value2 = (green[i][j] - nir[i][j]) / (green[i][j] + nir[i][j])
        arr1.append(value1)
        arr2.append(value2)
    ndviValueList.append(arr1)
    ndwiValueList.append(arr2)

ndviArr = numpy.array(ndviValueList)
ndwiArr = numpy.array(ndwiValueList)
print("ndvi")
print(ndviArr)
print("ndwi")
print(ndwiArr)

#XUẤT FILE TIFF
#need driver, height, width, count, dtype, crs, transform

#NDVI FILE
ndviList = []
ndviList.append(ndviArr)
ndviTiff = numpy.array(ndviList) #numpy array gồm 3 dimensions

dim1 = ndviTiff.shape
driver = "GTiff"
height = dim1[1]
width = dim1[2]
count = 1
dtype = ndviTiff.dtype
from rasterio.crs import CRS
crs = CRS.from_epsg(3424)
from rasterio.transform import from_origin
transform = from_origin(106356.577053, 1070273.49075, 26.949459, 26.949459) #tham số: left, right, x, y

with rasterio.open("ndvi.tif", "w", driver = driver, height= height, width = width, count = count, dtype = dtype,
                   crs = crs, transform = transform) as dst:
    dst.write(ndviTiff)
print("done")

#NDWI FILE
ndwiList = []
ndwiList.append(ndwiArr)
ndwiTiff = numpy.array(ndwiList)

dim2 = ndwiTiff.shape
driver = "GTiff"
height = dim2[1]
width = dim2[2]
count = 1
dtype = ndwiTiff.dtype
from rasterio.crs import CRS
crs = CRS.from_epsg(3424)

from rasterio.transform import from_origin
transform = from_origin(106356.577053, 1070273.49075, 26.949459, 26.949459) #tham số: left, right, x, y
with rasterio.open("ndwi.tif", "w", driver = driver, height= height, width = width, count = count, dtype = dtype,
                   crs = crs, transform = transform) as dst:
    dst.write(ndwiTiff)
print("done")