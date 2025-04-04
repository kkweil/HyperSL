import os
import h5pickle as h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from osgeo import gdal
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import time
from sklearn import preprocessing
import xml.etree.ElementTree as ET


class NoImplementException(Exception):
    def __init__(self, message):
        self.message = message

# TODO: 增加编辑功能、增添数据源、增添新数据
class MultisourceHSI:
    def __init__(self, root=None, dest=None, scale=preprocessing.minmax_scale):
        self.length = None
        self.datasets = None
        self.root = root
        self.scale = scale
        self.dest = dest
        self.length_intervals = None
        if root:
            self.datamap = self.creat_structure()
            # self.datamap = {}

    def create(self):
        with h5py.File(self.dest, "w") as f:
            for key in self.datamap.keys():
                g = f.create_group(key)
                for item in self.datamap[key]:
                    try:

                        if key == 'GF5':
                            hsi = self.read_GF5(item)
                        elif key == 'ENMap':
                            print(f'Adding HSI Product: {item.split(os.sep)[-2]}')
                            hsi, wavelength = self.read_EnMap(item)
                            # wavelength = self.readcenterwavelength(item + 'METADATA.xml')
                        elif key == 'DESIS':
                            print(f'Adding HSI Product: {item.split(os.sep)[-2]}')
                            hsi, wavelength = self.read_DESIS(item)
                            # wavelength = self.readcenterwavelength(item+'METADATA.xml')
                            # pass
                        else:
                            raise NoImplementException(f'Datasource: {key} is not implemented.')
                        if hsi:
                            d = g.create_dataset(item.split(os.sep)[-2], data=hsi)
                            # d.attrs['shape'] = hsi.shape
                            d.attrs['length'] = len(hsi)
                            d.attrs['bands'] = len(hsi[0])
                            d.attrs['wavelength'] = wavelength.astype('float32')
                    except NoImplementException as e:
                        e.message()

    def read(self, source='GF5'):
        f = h5py.File(self.dest, "r")
        g = f[source]
        self.datasets = [g[key] for key in g.keys()]
        self.length = sum([d.attrs['length'] for d in self.datasets])
        start = 0
        end = 0
        length_intervals = []
        lengths = [0] + [g[key].attrs['length'] for key in g.keys()]
        for i in range(len(lengths) - 1):
            start = start + lengths[i]
            end = end + lengths[i + 1]
            length_intervals.append((start, end - 1))
        self.length_intervals = length_intervals

    def creat_structure(self):
        dataset = {}
        for root, dirs, files in os.walk(self.root):
            level = root.replace(self.root, '').count(os.sep)
            if level == 1:
                dataset[root.replace(self.root + os.sep, '')] = []
            if level == 2:
                dataset[root.split(os.sep)[3]].append(os.path.join(root, files[0].replace(files[0].split('-')[-1], '')))
                # pass
            # for f in files:
            #     dataset[root.replace(self.root + os.sep, '')].append(os.path.join(root, f))
        return dataset

    def read_GF5(self, path):
        ruins_bands = list(range(192 - 1, 202)) + list(range(246 - 1, 262)) + list(range(327 - 1, 330))
        data = gdal.Open(path)
        width = data.RasterXSize
        height = data.RasterYSize
        hsi = data.ReadAsArray(0, 0, width, height)
        hsi = np.delete(hsi, ruins_bands, axis=0)
        shape = hsi.shape
        hsi = hsi.reshape(hsi.shape[0], -1).transpose()
        hsi = self.scale(hsi).transpose().reshape(-1, shape[1], shape[2]).transpose(1, 2, 0)
        return hsi

    def read_EnMap(self, path):
        NoData = -32768
        scale_coefficient = 0.0001
        ruins_bands = list(range(128, 135))
        wavelength = self.readcenterwavelength(path + 'METADATA.xml')
        dataset = gdal.Open(path + 'SPECTRAL_IMAGE.tif')
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        hsi = dataset.ReadAsArray(0, 0, width, height)
        if hsi is None:
            print(path + 'SPECTRAL_IMAGE.tif' + ' read failed.')
            return None, None
        else:
            hsi = hsi * scale_coefficient
            wavelength = np.array([float(n) for i, n in enumerate(wavelength) if i not in ruins_bands])
            hsi = np.delete(hsi, ruins_bands, axis=0)
            hsi[hsi == NoData * scale_coefficient] = np.nan
            shape = hsi.shape  # c,w,h
            hsi = hsi.reshape(hsi.shape[0], -1).transpose()
            hsi = self.scale(hsi).transpose().reshape(-1, shape[1], shape[2]).transpose(1, 2, 0)
            hsi = self.removeNan(hsi)
            return hsi, wavelength

    def read_DESIS(self, path):
        NoData = -32768
        scale_coefficient = 0.0001
        ruins_bands = list(range(0, 6))
        wavelength = self.readcenterwavelength(path + 'METADATA.xml')
        dataset = gdal.Open(path + 'SPECTRAL_IMAGE.tif')
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        hsi = dataset.ReadAsArray(0, 0, width, height)
        if hsi is None:
            print(path + 'SPECTRAL_IMAGE.tif' + ' read failed.')
            return None, None
        else:
            hsi = hsi * scale_coefficient
            wavelength = np.array([float(n) for i, n in enumerate(wavelength) if i not in ruins_bands])
            hsi = np.delete(hsi, ruins_bands, axis=0)
            hsi[hsi == NoData * scale_coefficient] = np.nan
            shape = hsi.shape  # c,w,h
            hsi = hsi.reshape(hsi.shape[0], -1).transpose()
            hsi = self.scale(hsi).transpose().reshape(-1, shape[1], shape[2]).transpose(1, 2, 0)
            hsi = self.removeNan(hsi)
            return hsi, wavelength

    @staticmethod
    def removeNan(data):
        mask = np.isnan(np.sum(data, axis=-1))
        value_idx = np.where(mask == False)
        data = [data[i, j].astype('float32') for i, j in zip(value_idx[0], value_idx[1])]
        return data

    @staticmethod
    def findcounts(img):
        shape = img.shape
        x1, y1 = 0, 0  # up left
        x2, y2 = 0, 0  # down right
        for i in range(shape[0]):
            if img[i, :].sum() != 0:
                for j in range(shape[1]):
                    if img[i, j] != 0:
                        y1 = j
                        break
                break

        for i in range(shape[1]):
            if img[:, i].sum() != 0:
                for j in range(shape[0]):
                    if img[j, i] != 0:
                        x1 = j
                        break
                break

        for i in range(shape[0]):
            if img[-(i + 1), :].sum() != 0:
                for j in range(shape[1]):
                    if img[-(i + 1), j] != 0:
                        y2 = j
                        break
                break

        for i in range(shape[1]):
            if img[:, -(i + 1)].sum() != 0:
                for j in range(shape[0]):
                    if img[j, -(i + 1)] != 0:
                        x2 = j
                        break
                break
        return x1, x2, y1, y2

    def readcenterwavelength(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        bandCharacterisation = root.find('specific').find('bandCharacterisation')
        return [child.find('wavelengthCenterOfBand').text for child in bandCharacterisation]


class MultisourceHSIDataset(Dataset):
    def __init__(self, mhsi_instance, source='GF5'):
        if not isinstance(mhsi_instance, MultisourceHSI):
            raise TypeError("The parameter 'mhsi_instance' must be an instance of the MultisourceHSI class.")

        mhsi_instance.read(source)
        self.datasets = mhsi_instance.datasets
        self.length = mhsi_instance.length
        self.length_intervals = mhsi_instance.length_intervals

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # temp = self.finddatasetidx(107595408)
        dataset_idx = self.finddatasetidx(idx)
        idx = idx - self.length_intervals[dataset_idx][0]
        # shape_axis0 = self.datasets[dataset_idx].attrs['shape'][0]
        # return dataset[idx % shape_axis0, idx // shape_axis0]
        return self.datasets[dataset_idx][idx][np.newaxis, :].astype('float32'), self.datasets[dataset_idx].attrs['wavelength'].astype('float32')

    def finddatasetidx(self, idx):
        low = 0
        high = len(self.length_intervals) - 1
        while low <= high:
            mid = (low + high) // 2
            start, end = self.length_intervals[mid]

            if start <= idx <= end:
                return mid  # 找到了数字所在的区间
            elif idx < start:
                high = mid - 1
            else:
                low = mid + 1
        return -1


if __name__ == '__main__':
    MHSIs = MultisourceHSI(root=r'E:\data\test', dest='../MultiSourceHSI_test.hdf5')
    MHSIs.create()
    # hsi = MHSIs.read_DESIS(
    #     r'../clfdata\DESIS\DESIS-HSI-L2A-DT0625231248_007-20230625T124701-V0220\DESIS-HSI-L2A-DT0625231248_007-20230625T124701-V0220-')
    # img_rgb = hsi[:, :, :3]
    # plt.imshow(img_rgb)
    # plt.show()
    # GF5_dataset = MultisourceHSIDataset(MHSIs, source='GF5')
    # EnMap_dataset = MultisourceHSIDataset(MHSIs, source='ENMap')
    # DESIS_dataset = MultisourceHSIDataset(MHSIs, source='DESIS')
    # EnMap_dataset.__getitem__(10000)
    # GF5_dataloader = DataLoader(GF5_dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=0)
    # EnMap_dataloader = DataLoader(EnMap_dataset, batch_size=4096, shuffle=True, pin_memory=True, num_workers=0)
    # DESIS_dataloader = DataLoader(DESIS_dataset, batch_size=4096, shuffle=True, pin_memory=True, num_workers=0)
    #

    # GF5_dataloader_cycle = cycle(GF5_dataloader)
    # EnMap_dataloader_cycle = cycle(EnMap_dataloader)
    # DESIS_dataloader_cycle = cycle(DESIS_dataloader)
    # n = 0
    # start_time = time.time()
    # for (enmap_s, enmap_w), (desis_s, desis_w) in zip(EnMap_dataloader, DESIS_dataloader):
    #     n += 1
    #     print(f'reading {n}')
    #     if torch.isnan(torch.sum(enmap_s)) or torch.isnan(torch.sum(desis_s)):
    #         print('Nan Data')
    #         break
    # end_time = time.time()
    # print(f'{end_time - start_time}s')  # 8 worker 5319s, 16 worker 5392s
    # print(max(len(EnMap_dataloader), len(DESIS_dataloader)))

    # en_iter=iter(EnMap_dataloader)
    # de_iter=iter(DESIS_dataloader)
    # start_time = time.time()
    # for _ in range(min(len(EnMap_dataloader), len(DESIS_dataloader))):
    #     en_S = EnMap_dataloader.__iter__().__next__()
    #     de_S = DESIS_dataloader.__iter__().__next__()
    #     n += 1
    #     print(f'reading {n}')
    # end_time = time.time()
    # print(f'{end_time - start_time}s')  # 292.146680355072s
    a = 0
