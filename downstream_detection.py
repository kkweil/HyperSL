from colorama import Fore
import warnings
from collections import OrderedDict
# from resconstruct_test import model

# warnings.filterwarnings("ignore")
# import matplotlib
# matplotlib.use('TkAgg')

from engine.changedection import ChangedetectionModel
from torch.utils.data import Dataset, DataLoader, Sampler
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
# import re

class CD_Dataset(Dataset):
    def __init__(self, data, label, wave):
        self.data = data
        self.label = label
        self.wave = wave

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32), self.wave.astype(np.float32), self.label[idx].astype(np.longlong)


def read_Hermiston(t1_path, t2_path, gt, waves, discard,):

    t1 = loadmat(t1_path)['HypeRvieW']
    t2 = loadmat(t2_path)['HypeRvieW']
    gt = loadmat(gt)['gt5clasesHermiston']

    wavelengths = []

    with open(discard,'r',encoding='utf-8') as d:
        lines =  d.readlines()
        for line in lines:
            if 'Bands to discard image1:' in line:
                t1_disc_idx = line.split(': ')[-1]
                t1_disc_idx = [int(i) for i in t1_disc_idx.split(' ')[:-1]]
            if 'Bands to discard image2:' in line:
                t2_disc_idx = line.split(': ')[-1]
                t2_disc_idx = [int(i) for i in t2_disc_idx.split(' ')[:-1]]

    disc_idx = list(set(t1_disc_idx)|set(t2_disc_idx))

    with open(waves,'r') as w:
        lines = w.readlines()
        for line in lines:
            wavelengths.append(float(line.split('=')[-1][1:-2]))

    wavelengths = np.asarray(wavelengths)
    wavelengths = np.delete(wavelengths,disc_idx)


    t1 = np.delete(t1,disc_idx,axis=-1)
    t2 = np.delete(t2, disc_idx,axis=-1)
    W, H, C = t1.shape
    t1 = t1.reshape(-1, C)
    t1 = minmax_scale(t1)
    t1 = t1.reshape(W, H, C)

    t2 = t2.reshape(-1, C)
    t2 = minmax_scale(t2)
    t2 = t2.reshape(W, H, C)

    return t1,t2, wavelengths,gt

def read_bayArea(t1_path,t2_path,gt_path,wave_path,):

    t1 = loadmat(t1_path)['HypeRvieW']
    t2 = loadmat(t2_path)['HypeRvieW']
    gt = loadmat(gt_path)['HypeRvieW']
    W, H, C = t1.shape
    wavelengths = []
    with open(wave_path,'r') as w:
        lines = w.readlines()
        for line in lines:
            wavelengths.append(float(line.split(' ')[3]))
    # u = np.mean(t1,axis=(0,1))
    t1 = t1.reshape(-1, C)
    t1 = minmax_scale(t1)
    t1 = t1.reshape(W, H, C)

    t2 = t2.reshape(-1, C)
    t2 = minmax_scale(t2)
    t2 = t2.reshape(W, H, C)


    return t1,t2, np.asarray(wavelengths),gt

def read_santaBarbara(t1_path,t2_path,gt_path,wave_path,):

    t1 = loadmat(t1_path)['HypeRvieW']
    t2 = loadmat(t2_path)['HypeRvieW']
    gt = loadmat(gt_path)['HypeRvieW']
    W, H, C = t1.shape
    wavelengths = []
    with open(wave_path,'r') as w:
        lines = w.readlines()
        for line in lines:
            # print(line.split(' '))
            wavelengths.append(float(line.split(' ')[3]))
    # u = np.mean(t1,axis=(0,1))
    t1 = t1.reshape(-1, C)
    t1 = minmax_scale(t1)
    t1 = t1.reshape(W, H, C)

    t2 = t2.reshape(-1, C)
    t2 = minmax_scale(t2)
    t2 = t2.reshape(W, H, C)

    return t1,t2, np.asarray(wavelengths),gt

def split_train_test(t1, t2, label, patch_size, ratio):
    assert patch_size % 2 == 1, 'The window size must be odd.'
    R = int(patch_size / 2)
    label = np.pad(label, R, mode='constant')
    t1 = np.pad(t1, (R, R), mode='reflect')[:, :, R:-R]
    t2 = np.pad(t2, (R, R), mode='reflect')[:, :, R:-R]
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

        x_train_t1 = np.asarray([t1[coordinate_train[i][0] - R: coordinate_train[i][0] + R + 1,
                              coordinate_train[i][1] - R: coordinate_train[i][1] + R + 1, :]
                              for i in range(len(coordinate_train))])
        x_train_t2 = np.asarray([t2[coordinate_train[i][0] - R: coordinate_train[i][0] + R + 1,
                              coordinate_train[i][1] - R: coordinate_train[i][1] + R + 1, :]
                              for i in range(len(coordinate_train))])
        y_train = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_train])

        x_test_t1 = np.asarray([t1[coordinate_test[i][0] - R: coordinate_test[i][0] + R + 1,
                             coordinate_test[i][1] - R: coordinate_test[i][1] + R + 1, :]
                             for i in range(len(coordinate_test))])
        x_test_t2 = np.asarray([t2[coordinate_test[i][0] - R: coordinate_test[i][0] + R + 1,
                             coordinate_test[i][1] - R: coordinate_test[i][1] + R + 1, :]
                             for i in range(len(coordinate_test))])
        y_test = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_test])

        x_train = np.concatenate((x_train_t1[:, np.newaxis], x_train_t2[:, np.newaxis]), axis=1)
        x_test = np.concatenate((x_test_t1[:, np.newaxis], x_test_t2[:, np.newaxis]), axis=1)

        return x_train, x_test, y_train, y_test, coordinate_train, coordinate_test

    elif isinstance(ratio, float):
        print(f"Sample the training dataset at a scale of {ratio}")
        coordinate_train = [v[:math.ceil(len(v) * ratio)].tolist() for k, v in coordinate.items()]
        coordinate_train = np.asarray([i for list_ in coordinate_train for i in list_])

        coordinate_test = [v[math.ceil(len(v) * ratio):].tolist() for k, v in coordinate.items()]
        coordinate_test = np.asarray([i for list_ in coordinate_test for i in list_])

        x_train_t1 = np.asarray([t1[coordinate_train[i][0] - R: coordinate_train[i][0] + R + 1,
                                 coordinate_train[i][1] - R: coordinate_train[i][1] + R + 1, :]
                                 for i in range(len(coordinate_train))])
        x_train_t2 = np.asarray([t2[coordinate_train[i][0] - R: coordinate_train[i][0] + R + 1,
                                 coordinate_train[i][1] - R: coordinate_train[i][1] + R + 1, :]
                                 for i in range(len(coordinate_train))])
        y_train = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_train])

        x_test_t1 = np.asarray([t1[coordinate_test[i][0] - R: coordinate_test[i][0] + R + 1,
                                coordinate_test[i][1] - R: coordinate_test[i][1] + R + 1, :]
                                for i in range(len(coordinate_test))])
        x_test_t2 = np.asarray([t2[coordinate_test[i][0] - R: coordinate_test[i][0] + R + 1,
                                coordinate_test[i][1] - R: coordinate_test[i][1] + R + 1, :]
                                for i in range(len(coordinate_test))])
        y_test = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_test])

        x_train = np.concatenate((x_train_t1[:,np.newaxis], x_train_t2[:,np.newaxis]),axis=1)
        x_test = np.concatenate((x_test_t1[:,np.newaxis], x_test_t2[:,np.newaxis]),axis=1)

        return x_train, x_test, y_train, y_test, coordinate_train, coordinate_test

    else:
        raise ValueError('Expect "ratio" for: int or float!')


def calculate_F1PrRe(CM):
    # 提取混淆矩阵中的 TP, FP, FN, TN
    TP = CM[0, 0]
    FP = CM[0, 1]
    FN = CM[1, 0]
    TN = CM[1, 1]

    # Precision（精度）: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0

    # Recall（召回率）: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return precision, recall, f1_score

if __name__ == '__main__':

    dataset_name = 'BayArea'

    
    if dataset_name == 'BayArea':

        t1, t2, waves, gt = read_bayArea('CD_Dataset/bayArea_AVIRIS/mat/Bay_Area_2013.mat',
                                          'CD_Dataset/bayArea_AVIRIS/mat/Bay_Area_2015.mat',
                                          'CD_Dataset/bayArea_AVIRIS/mat/bayArea_gtChanges2.mat.mat',
                                          'CD_Dataset/bayArea_AVIRIS/waves.txt',
                                          )
    if dataset_name == 'Barbara': 
        t1,t2, waves,gt = read_santaBarbara('CD_Dataset/santaBarbara_AVIRIS/mat/barbara_2013.mat',
                               'CD_Dataset/santaBarbara_AVIRIS/mat/barbara_2014.mat',
                               'CD_Dataset/santaBarbara_AVIRIS/mat/barbara_gtChanges.mat',
                               'CD_Dataset/santaBarbara_AVIRIS/waves.txt',
                               )
    #
    # t1,t2, waves,gt =read_Hermiston('CD_Dataset/Hermiston/hermiston2004.mat',
    #                                              'CD_Dataset/Hermiston/hermiston2007.mat',
    #                                              'CD_Dataset/Hermiston/rdChangesHermiston_5classes.mat',
    #                                              'CD_Dataset/Hermiston/reduced/EO1_HyperSpectral_metadata.txt',
    #                                              'CD_Dataset/Hermiston/reduced/discarted_output.txt')
    # rgb1 = np.zeros((390,200,3))
    # rgb2 = np.zeros((390, 200, 3))
    #
    # rgb1[:,:,0] =t1[:,:,30]
    # rgb1[:, :, 1] =t1[:,:,20]
    # rgb1[:, :, 2] =t1[:,:,10]
    #
    # rgb2[:,:,0] =t2[:,:,30]
    # rgb2[:, :, 1] =t2[:,:,20]
    # rgb2[:, :, 2] =t2[:,:,10]
    # plt.imshow(rgb1)
    # plt.show()

    x_train, x_test, y_train, y_test, coordinate_train, coordinate_test = split_train_test(t1, t2, gt, 7, 0.01)
    x_all = np.concatenate([x_train,x_test])
    y_all = np.concatenate([y_train,y_test])
    coordinate_all = np.concatenate([coordinate_train, coordinate_test])
    cd_train_set = CD_Dataset(x_train,y_train,waves)
    cd_loader = DataLoader(cd_train_set,batch_size=6,shuffle=True,drop_last=True)

    cd_test_set = CD_Dataset(x_test,y_test,waves)
    cd_loader_test = DataLoader(cd_test_set,batch_size=64,shuffle=True)

    Dataset_all = CD_Dataset(x_all, y_all, waves)
    Loader_all = DataLoader(Dataset_all, batch_size=64, shuffle=False)

    model = ChangedetectionModel(class_num = 2,
                                 model_size='small')

    # ckpt = torch.load('10_base_mask95_checkpoint.pt')
    # weights = OrderedDict()
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
    f = open(f'{dataset_name}.txt','w')
    for epoch in range(100):
        loss_ = []
        correct = 0
        model.train()
        for x,w,y in cd_loader:
            # print('x:',x[:,0,:].shape)
            # print('w:',w.shape)
            # print('y:', y.shape)
            logits = model(x[:,0,:].cuda(),x[:,1,:].cuda(),w.cuda(),w.cuda())
            loss = entropy(logits,y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                # print(loss.item())
            loss_.append(loss.item())
            correct = correct + (logits.argmax(1) == y.cuda()).sum().item()

        print(Fore.RESET + f'Epoch{epoch + 1}: mean_loss:{sum(loss_) / len(loss_)} acc:{correct / len(x_train)}', file=f, flush=True)

        if (epoch + 1) % 100 == 0:
            model.eval()
            preds = np.empty(shape=(0,))
            labels = np.empty(shape=(0,))
            # print('ttt')
            with torch.no_grad():

                for x, w, y in cd_loader_test:
                        # print(model(x.cuda(), w.cuda()).argmax(-1).cpu().numpy().shape)
                    # if w.shape[0] != 80 :
                    #     print('1111')
                    preds = np.concatenate([preds, model(x[:,0,:].cuda(),x[:,1,:].cuda(),w.cuda(),w.cuda()).argmax(-1).cpu().numpy()])
                    labels = np.concatenate([labels, y.cpu().numpy()])
                        # print(f'x:{x.shape},w:{w.shape},y:{y.shape}')
                preds = preds.astype('int')
                labels = labels.astype('int')
                cls_nums = {}
                for i in range(labels.max()):
                    cls_nums[i] = sum(labels == i)
                from sklearn.metrics import confusion_matrix

                CM = confusion_matrix(labels, preds)
                aa = []
                for i in range(labels.max()):
                    aa.append(CM[i, i] / cls_nums[i])
                AA = sum(aa) / len(aa)

                l = preds - labels
                OA = sum(l == 0) / len(l)

                k = 0
                for i in range(labels.max()):
                    c = sum(CM[i, :])
                    r = sum(CM[:, i])
                    k = k + r * c
                k = k / (len(preds) ** 2)
                Kappa = (OA - k) / (1 - k)

                Pr, Re, F1 = calculate_F1PrRe(CM)

                print(Fore.RED + f'[TEST] Epoch{epoch + 1}: OA:{OA} AA:{AA} Kappa:{Kappa},Pr:{Pr} Re:{Re} F1:{F1}', file=f, flush=True)

    with torch.no_grad():
        preds = np.empty(shape=(0,))
        labels = np.empty(shape=(0,))
        model.eval()
        with torch.no_grad():
            for x, w, y in Loader_all:
                # print(model(x.cuda(), w.cuda()).argmax(-1).cpu().numpy().shape)
                preds = np.concatenate([preds, model(x[:,0,:].cuda(),x[:,1,:].cuda(),w.cuda(),w.cuda()).argmax(-1).cpu().numpy()])
                labels = np.concatenate([labels, y.cpu().numpy()])
                # print(f'x:{x.shape},w:{w.shape},y:{y.shape}')
            preds = preds.astype('int')
            labels = labels.astype('int')
            predict_img = np.full((gt.shape[0], gt.shape[1], 3), 127)
            predict_img = np.pad(predict_img, 3)[:,:,3:-3]

            for idx, item in enumerate(labels):
                if item == preds[idx]:
                    pass
                elif item != preds[idx] and item == 1:
                    preds[idx] = 3  # FP
                elif item != preds[idx] and item == 0:
                    preds[idx] = 4  # FN

            for idx, p in enumerate(coordinate_all):
                x = p[0]
                y = p[1]
                v = preds[idx]
                if v == 0:
                    predict_img[x,y] = [255, 255, 255]
                if v == 1:
                    predict_img[x,y] =  [0, 0, 0]
                if v == 3:
                    predict_img[x,y] = [255, 0, 0]
                if v == 4:
                    predict_img[x,y] = [0, 255, 0]

            plt.imshow(predict_img[3:-3, 3:-3])
            plt.axis('off')
            plt.imsave('bayArea_pred.png', predict_img[3:-3, 3:-3].astype('uint8'))
            # plt.show()
    a = 0