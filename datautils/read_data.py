# import numpy as np
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib
import numpy.ma as ma
from sklearn import preprocessing

matplotlib.use('TkAgg')


def read_EnMap(path, NoData, ruins_bands, scale):
    dataset = gdal.Open(path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    hsi = dataset.ReadAsArray(0, 0, width, height) * scale_coefficient
    hsi = np.delete(hsi, ruins_bands, axis=0)
    hsi[hsi == NoData * scale_coefficient] = np.nan
    shape = hsi.shape  # c,w,h
    hsi = hsi.reshape(hsi.shape[0], -1).transpose()
    hsi = scale(hsi).transpose().reshape(-1, shape[1], shape[2]).transpose(1, 2, 0)
    return hsi


def read_DESIS(path, NoData, ruins_bands, scale):
    dataset = gdal.Open(path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    hsi = dataset.ReadAsArray(0, 0, width, height)
    hsi = np.delete(hsi, ruins_bands, axis=0)
    hsi[hsi == NoData] = np.nan
    shape = hsi.shape  # c,w,h
    hsi = hsi.reshape(hsi.shape[0], -1).transpose()
    hsi = scale(hsi).transpose().reshape(-1, shape[1], shape[2]).transpose(1, 2, 0)
    return hsi


if __name__ == '__main__':
    NoData = -32768
    scale_coefficient = 0.0001
    # ruins_bands = list(range(0, 6))
    ruins_bands = list(range(130 - 1, 135))
    path = r'/clfdata\ENMAP\ENMap-HSI-L2A-DT0000001010_20220609T062650Z_008_V010402_20240704T054656Z\ENMAP01-____L2A-DT0000001010_20220609T062650Z_008_V010402_20240704T054656Z-SPECTRAL_IMAGE.TIF'
    im = read_DESIS(path, NoData, ruins_bands, preprocessing.minmax_scale)
    img_rgb = im[:, :, :3]
    plt.imshow(img_rgb)
    plt.show()
    a = 0
