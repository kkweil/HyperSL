import random

from engine.adptive_cnn import Encoder
from colorama import Fore
import warnings

warnings.filterwarnings("ignore")

from collections import OrderedDict
from engine.classification import ClassificationModel
from torch.utils.data import Dataset, DataLoader, Sampler
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import math


class HyperDataset(Dataset):
    def __init__(self, data, ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.cuda.is_available():
            return torch.from_numpy(self.data[idx]).float().cuda()


def read_indian_pines(data_path, label_path, wave_path):
    remove_bands = list(range(103, 108)) + list(range(149, 163)) + [119]
    hsi = loadmat(data_path)['input']
    gt = loadmat(label_path)['indian_pines_gt']
    W, H, C = hsi.shape
    hsi = hsi.reshape(-1, C)
    hsi = minmax_scale(hsi)
    hsi = hsi.reshape(W, H, C)
    waves = pd.read_csv(wave_path)
    waves = np.array(waves.iloc[:, 1])[:220]
    waves = np.delete(waves, remove_bands)

    return hsi, gt, waves


def read_botswana(data_path, label_path, wave_path):
    hsi = loadmat(data_path)['Botswana']
    gt = loadmat(label_path)['Botswana_gt']
    W, H, C = hsi.shape
    hsi = hsi.reshape(-1, C)
    hsi = minmax_scale(hsi)
    hsi = hsi.reshape(W, H, C)
    waves = pd.read_csv(wave_path)
    waves = np.array(waves.iloc[:, 1])

    return hsi, gt, waves


def split_image(image, patch_size=(32, 32), stride=(8, 32)):
    M, N, C = image.shape
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride

    # 计算出切分后的子图数量
    patches = []
    for i in range(0, M - patch_height + 1, stride_height):
        for j in range(0, N - patch_width + 1, stride_width):
            patch = image[i:i + patch_height, j:j + patch_width]
            patches.append(patch)

    # 将所有子图拼接成一个大的张量，形状为 (num_patches, patch_height, patch_width)
    # patches_tensor = torch.stack(patches)  # [num_patches, patch_height, patch_width]

    return patches


if __name__ == '__main__':
    data_bos, label, wavelength = read_botswana('clfdata/Botswana/Botswana.mat',
                                                'clfdata/Botswana/Botswana_gt.mat',
                                                'clfdata/Botswana/wave.csv')

    data_bos = split_image(data_bos, patch_size=(32, 32), stride=(8, 32))
    random.shuffle(data_bos)
    train_set_bos = HyperDataset(data_bos[:math.ceil(0.8 * len(data_bos))])
    valid_set_bos = HyperDataset(data_bos[math.ceil(0.8 * len(data_bos)):math.ceil(0.9 * len(data_bos))])
    test_set_bos = HyperDataset(data_bos[math.ceil(0.8 * len(data_bos)):])

    train_bos_dataloader = DataLoader(train_set_bos, batch_size=32, shuffle=True)
    valid_bos_dataloader = DataLoader(valid_set_bos, batch_size=32, shuffle=True)
    test_bos_dataloader = DataLoader(test_set_bos, batch_size=32, shuffle=True)

    data_IN, _, _ = read_indian_pines('clfdata/Botswana/Botswana.mat',
                                      'clfdata/Botswana/Botswana_gt.mat',
                                      'clfdata/Botswana/wave.csv')

    data_IN = split_image(data_IN, patch_size=(32, 32), stride=(8, 32))
    random.shuffle(data_IN)
    train_set_IN = HyperDataset(data_IN[:math.ceil(0.8 * len(data_IN))])
    valid_set_IN = HyperDataset(data_IN[math.ceil(0.8 * len(data_IN)):math.ceil(0.9 * len(data_IN))])
    test_set_IN = HyperDataset(data_IN[math.ceil(0.8 * len(data_IN)):])

    train_IN_dataloader = DataLoader(train_set_IN, batch_size=32, shuffle=True)
    valid_IN_dataloader = DataLoader(valid_set_IN, batch_size=32, shuffle=True)
    test_IN_dataloader = DataLoader(test_set_IN, batch_size=32, shuffle=True)

    model = Encoder(
        embed_dim=108,
        hidden_dim=32,
        output_dim=400,
        img_size=32
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, )
    # loss
    mesloss = torch.nn.MSELoss()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(100):
        loss_train = []
        loss_valid = []
        correct = 0
        model.train()
        for x in train_dataloader:
            B, H, W, C = x.shape
            y_pred, _ = model(x)
            loss = mesloss(y_pred[..., :C], x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())

        with torch.no_grad():
            model.eval()
            for x in valid_dataloader:
                B, H, W, C = x.shape
                y_pred, _ = model(x)
                loss = mesloss(y_pred[..., :C], x)
                loss_valid.append(loss.item())

        print(
            Fore.RESET + f'Epoch{epoch + 1}: train_loss:{sum(loss_train) / len(loss_train)} valid_loss:{sum(loss_valid) / len(loss_valid)}')

        if (epoch + 1) % 20 == 0:
            loss_test = []
            model.eval()
            with torch.no_grad():
                for x in test_dataloader:
                    B, H, W, C = x.shape
                    y_pred, _ = model(x)
                    loss = mesloss(y_pred[..., :C], x)
                    loss_test.append(loss.item())
                print(Fore.RESET + f'Epoch{epoch + 1}: test_loss:{sum(loss_test) / len(loss_test)}')
