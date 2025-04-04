import pickle as plk
import numpy as np
import os
import uuid

import math
import torch.utils.data
from sklearn.preprocessing import scale, minmax_scale


def get_dataarchive(path):
    data_archive = {'clfdata': {}, 'gt': {}}
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'gt' not in file.split('_')[-1]:
                with open(os.path.join(root, file), 'rb') as d:
                    data_archive['clfdata'][file.split('.')[0]] = plk.load(d)
                # clfdata = plk.load(d)
            else:
                with open(os.path.join(root, file), 'rb') as d:
                    data_archive['gt'][file.split('.')[0]] = plk.load(d)
    channels = []
    for key, item in data_archive['clfdata'].items():
        if key.split('_')[-1] != 'gt':
            channels.append(key.split('_')[-1])
    return data_archive, set(channels)


def get_spectra(**kargs):
    dataset = []
    # shapes = []
    for key, data in kargs['clfdata'].items():
        shape = data.shape  # w,h,c
        # shapes.append(shape)
        data = data.reshape(-1, shape[-1])
        # clfdata = scale(clfdata).astype('float32')
        data = minmax_scale(data).astype('float32')
        data = np.expand_dims(data, axis=1)
        dataset.append((data, shape))
    return dataset


def ReflectionPaddding(x, bands=1, Cr=32):
    delta = bands - math.floor(bands / Cr) * Cr
    return x[:, :bands-delta]


class SpectraDataset(torch.utils.data.Dataset):
    def __init__(self, data, shape, Cr):
        self.data = data
        self.shape = shape
        self.Cr = Cr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return ReflectionPaddding(self.clfdata[idx], bands=self.shape[-1], Cr=32)
        return self.data[idx]


if __name__ == '__main__':
    data_archive, _ = get_dataarchive('../clfdata')
    data = get_spectra(**data_archive)
    datasets = [SpectraDataset(subdata) for subdata in data]
    loaders = [torch.utils.data.DataLoader(subset, batch_size=32) for subset in datasets]

    for loader in loaders:
        for X in loader:
            print(X.shape)

    a = 1
