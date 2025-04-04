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
    def __init__(self, data, label, wave):
        self.data = data
        self.label = label
        self.wave = wave

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32), self.wave.astype(np.float32), self.label[idx].astype(np.longlong)


def split_train_test(data, label, patch_size, ratio):
    assert patch_size % 2 == 1, 'The window size must be odd.'
    R = int(patch_size / 2)
    label = np.pad(label, R, mode='constant')
    data = np.pad(data, (R, R), mode='reflect')[:, :, R:-R]
    coordinate = {i: np.argwhere(label == i + 1) for i in range(0, label.max())}
    for _, v in coordinate.items():
        np.random.shuffle(v)

    if isinstance(ratio, int):
        print(f"Sample the training dataset {ratio} points per class.")
        coordinate_train = []
        coordinate_test = []
        for k, v in coordinate.items():
            if len(v) > 2 * ratio:
                coordinate_train.append(v[:ratio].tolist())
                coordinate_test.append(v[ratio:].tolist())
            else:
                coordinate_train.append(v[:int(len(v) / 2)].tolist())
                coordinate_test.append(v[int(len(v) / 2):].tolist())

        coordinate_train = np.asarray([i for list_ in coordinate_train for i in list_])
        coordinate_test = np.asarray([i for list_ in coordinate_test for i in list_])

        x_train = np.asarray([data[coordinate_train[i][0] - R: coordinate_train[i][0] + R + 1,
                              coordinate_train[i][1] - R: coordinate_train[i][1] + R + 1, :]
                              for i in range(len(coordinate_train))])
        y_train = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_train])

        x_test = np.asarray([data[coordinate_test[i][0] - R: coordinate_test[i][0] + R + 1,
                             coordinate_test[i][1] - R: coordinate_test[i][1] + R + 1, :]
                             for i in range(len(coordinate_test))])
        y_test = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_test])

        return x_train, x_test, y_train, y_test, coordinate_train, coordinate_test

    elif isinstance(ratio, float):
        print(f"Sample the training dataset at a scale of {ratio}")
        coordinate_train = [v[:math.ceil(len(v) * ratio)].tolist() for k, v in coordinate.items()]
        coordinate_train = np.asarray([i for list_ in coordinate_train for i in list_])

        coordinate_test = [v[math.ceil(len(v) * ratio):].tolist() for k, v in coordinate.items()]
        coordinate_test = np.asarray([i for list_ in coordinate_test for i in list_])

        x_train = np.asarray([data[coordinate_train[i][0] - R: coordinate_train[i][0] + R + 1,
                              coordinate_train[i][1] - R: coordinate_train[i][1] + R + 1, :]
                              for i in range(len(coordinate_train))])
        y_train = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_train])

        x_test = np.asarray([data[coordinate_test[i][0] - R: coordinate_test[i][0] + R + 1,
                             coordinate_test[i][1] - R: coordinate_test[i][1] + R + 1, :]
                             for i in range(len(coordinate_test))])
        y_test = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_test])

        return x_train, x_test, y_train, y_test, coordinate_train, coordinate_test

    else:
        raise ValueError('Expect "ratio" for: int or float!')


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

    return hsi, gt,waves

if __name__ == '__main__':

    data, label,wavelength = read_botswana('clfdata/Botswana/Botswana.mat',
                                           'clfdata/Botswana/Botswana_gt.mat',
                                           'clfdata/Botswana/wave.csv')

    x_train, x_test, y_train, y_test, coordinate_train, coordinate_test = split_train_test(data, label, 15, 10)
    print(x_train.shape)

    Dataset_train = HyperDataset(x_train,y_train,wavelength)
    Loader_train = DataLoader(Dataset_train,batch_size=16,shuffle=True)

    Dataset_test = HyperDataset(x_test,y_test,wavelength)
    Loader_test = DataLoader(Dataset_test,batch_size=32,shuffle=True)

    model = ClassificationModel(
        class_num=16,
        model_size='small',
    )


    # ckpt = torch.load('10_base_mask95_checkpoint.pt')
    weights = OrderedDict()
    # for k, v in ckpt['model'].items():
    #     name = k[7:]
    #     weights[name] = v
    # model.spectral_encoder.load_state_dict(weights)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,)
    # loss
    entropy = torch.nn.CrossEntropyLoss()

    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model,device_ids=[0,1])


    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(100):
        loss_ = []
        correct = 0
        model.train()
        for x,w,y in Loader_train:
            y_pred = model(x.cuda(),w.cuda())
            loss = entropy(y_pred,y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.item())
            loss_.append(loss.item())
            correct = correct + (y_pred.argmax(1) == y.cuda()).sum().item()
        
        print(Fore.RESET + f'Epoch{epoch + 1}: mean_loss:{sum(loss_) / len(loss_)} acc:{correct / len(x_train)}')
        if (epoch+1) % 20 ==0:
            model.eval()
            preds = np.empty(shape=(0,))
            labels = np.empty(shape=(0,))
            with torch.no_grad():
                for x, w, y in Loader_test:
                    # print(model(x.cuda(), w.cuda()).argmax(-1).cpu().numpy().shape)
                    preds = np.concatenate([preds, model(x.cuda(), w.cuda()).argmax(-1).cpu().numpy()])
                    labels = np.concatenate([labels, y.cpu().numpy()])
                    # print(f'x:{x.shape},w:{w.shape},y:{y.shape}')
                preds = preds.astype

    # data, label,wavelength = read_botswana('clfdata/Botswana/Botswana.mat',
    #                                        'clfdata/Botswana/Botswana_gt.mat',
    #                                        'clfdata/Botswana/wave.csv')
    #
    # labels = labels.astype('int')
    # cls_nums = {}
    # for i in range(labels.max()):
    #     cls_nums[i] = sum(labels == i)
    # from sklearn.metrics import confusion_matrix
    #
    # CM = confusion_matrix(labels, preds)
    # aa = []
    # for i in range(labels.max()):
    #     aa.append(CM[i, i] / cls_nums[i])
    # AA = sum(aa) / len(aa)
    #
    # l = preds - labels
    # OA = sum(l == 0) / len(l)
    #
    # k = 0
    # for i in range(labels.max()):
    #     c = sum(CM[i, :])
    #     r = sum(CM[:, i])
    #     k = k + r * c
    # k = k / (len(preds) ** 2)
    # Kappa = (OA - k) / (1 - k)
    # print(Fore.RED + f'Epoch{epoch + 1}: OA:{OA} AA:{AA} Kappa:{Kappa}')