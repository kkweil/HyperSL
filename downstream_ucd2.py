from colorama import Fore
import argparse
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

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from sklearn.decomposition import PCA
# from resconstruct_test import model

# warnings.filterwarnings("ignore")
# import matplotlib
# matplotlib.use('TkAgg')


from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale
import numpy as np
import torch
import math
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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


def read_Hermiston(t1_path, t2_path, gt, waves, discard, ):
    t1 = loadmat(t1_path)['HypeRvieW']
    t2 = loadmat(t2_path)['HypeRvieW']
    gt = loadmat(gt)['gt5clasesHermiston']

    wavelengths = []

    with open(discard, 'r', encoding='utf-8') as d:
        lines = d.readlines()
        for line in lines:
            if 'Bands to discard image1:' in line:
                t1_disc_idx = line.split(': ')[-1]
                t1_disc_idx = [int(i) for i in t1_disc_idx.split(' ')[:-1]]
            if 'Bands to discard image2:' in line:
                t2_disc_idx = line.split(': ')[-1]
                t2_disc_idx = [int(i) for i in t2_disc_idx.split(' ')[:-1]]

    disc_idx = list(set(t1_disc_idx) | set(t2_disc_idx))

    with open(waves, 'r') as w:
        lines = w.readlines()
        for line in lines:
            wavelengths.append(float(line.split('=')[-1][1:-2]))

    wavelengths = np.asarray(wavelengths)
    wavelengths = np.delete(wavelengths, disc_idx)

    t1 = np.delete(t1, disc_idx, axis=-1)
    t2 = np.delete(t2, disc_idx, axis=-1)
    W, H, C = t1.shape
    t1 = t1.reshape(-1, C)
    t1 = minmax_scale(t1)
    t1 = t1.reshape(W, H, C)

    t2 = t2.reshape(-1, C)
    t2 = minmax_scale(t2)
    t2 = t2.reshape(W, H, C)

    return t1, t2, wavelengths, gt


def read_bayArea(t1_path, t2_path, gt_path, wave_path, ):
    t1 = loadmat(t1_path)['HypeRvieW']
    t2 = loadmat(t2_path)['HypeRvieW']
    gt = loadmat(gt_path)['HypeRvieW']
    W, H, C = t1.shape
    wavelengths = []
    with open(wave_path, 'r') as w:
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

    return t1, t2, np.asarray(wavelengths), gt


def read_santaBarbara(t1_path, t2_path, gt_path, wave_path, ):
    t1 = loadmat(t1_path)['HypeRvieW']
    t2 = loadmat(t2_path)['HypeRvieW']
    gt = loadmat(gt_path)['HypeRvieW']
    W, H, C = t1.shape
    wavelengths = []
    with open(wave_path, 'r') as w:
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

    return t1, t2, np.asarray(wavelengths), gt


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

        x_train = np.concatenate((x_train_t1[:, np.newaxis], x_train_t2[:, np.newaxis]), axis=1)
        x_test = np.concatenate((x_test_t1[:, np.newaxis], x_test_t2[:, np.newaxis]), axis=1)

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


def cluster_accuracy(true_labels, cluster_labels):
    """
    计算聚类结果的最佳匹配准确率 (Overall Accuracy, OA)
    :param true_labels: 真实标签 (ground truth)
    :param cluster_labels: 聚类结果的标签
    :return: (OA, best_mapping_labels)
    """
    assert len(true_labels) == len(cluster_labels)

    # 获取唯一类别
    unique_true_labels = np.unique(true_labels)
    unique_cluster_labels = np.unique(cluster_labels)

    # 构造混淆矩阵
    cost_matrix = np.zeros((len(unique_true_labels), len(unique_cluster_labels)))

    for i, t in enumerate(unique_true_labels):
        for j, c in enumerate(unique_cluster_labels):
            cost_matrix[i, j] = np.sum((true_labels == t) & (cluster_labels == c))

    # 使用匈牙利算法进行最佳类别匹配
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # 负号表示最大匹配

    # 构造映射关系 (cluster_label -> true_label)
    best_mapping = dict(zip(unique_cluster_labels[col_ind], unique_true_labels[row_ind]))
    mapped_preds = np.array([best_mapping[label] for label in cluster_labels])

    # 计算 Overall Accuracy (OA)
    oa = np.mean(mapped_preds == true_labels)

    return oa, mapped_preds


def compute_aa_kappa(true_labels, pred_labels):
    """
    计算 Average Accuracy (AA) 和 Kappa 系数
    :param true_labels: 真实标签
    :param pred_labels: 经过最佳匹配的聚类标签
    :return: (AA, Kappa)
    """
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # 计算每个类别的分类准确率
    class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

    # 计算 Average Accuracy (AA)
    aa = np.mean(class_accuracy)

    # 计算 Kappa 系数
    total_samples = np.sum(conf_matrix)
    p0 = np.sum(np.diag(conf_matrix)) / total_samples
    pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_samples ** 2)
    kappa = (p0 - pe) / (1 - pe)

    return aa, kappa


if __name__ == '__main__':
    # dataset_name = 'BayArea'
    #
    # if dataset_name == 'BayArea':
    #     t1, t2, waves, gt = read_bayArea('CD_Dataset/bayArea_AVIRIS/mat/Bay_Area_2013.mat',
    #                                      'CD_Dataset/bayArea_AVIRIS/mat/Bay_Area_2015.mat',
    #                                      'CD_Dataset/bayArea_AVIRIS/mat/bayArea_gtChanges2.mat.mat',
    #                                      'CD_Dataset/bayArea_AVIRIS/waves.txt',
    #                                      )
    # if dataset_name == 'Barbara':
    #     t1, t2, waves, gt = read_santaBarbara('CD_Dataset/santaBarbara_AVIRIS/mat/barbara_2013.mat',
    #                                           'CD_Dataset/santaBarbara_AVIRIS/mat/barbara_2014.mat',
    #                                           'CD_Dataset/santaBarbara_AVIRIS/mat/barbara_gtChanges.mat',
    #                                           'CD_Dataset/santaBarbara_AVIRIS/waves.txt',
    #                                           )
    # x_train, x_test, y_train, y_test, coordinate_train, coordinate_test = split_train_test(t1, t2, gt, 7, 0.1)
    # x_all = np.concatenate([x_train, x_test])
    # y_all = np.concatenate([y_train, y_test])
    # coordinate_all = np.concatenate([coordinate_train, coordinate_test])
    #
    # Dataset_all = CD_Dataset(x_all, y_all, waves)
    # Loader_all = DataLoader(Dataset_all, batch_size=64, shuffle=False)
    #
    # model = ChangedetectionModel(class_num=2,
    #                              model_size='small')
    #
    # ckpt = torch.load('10_base_mask95_checkpoint.pt')
    # weights = OrderedDict()
    # for k, v in ckpt['model'].items():
    #     name = k[7:]
    #     weights[name] = v
    # model.spectral_encoder.load_state_dict(weights)
    #
    # if torch.cuda.is_available():
    #     model.cuda()
    #
    # with torch.no_grad():
    #     hiddens = np.empty(shape=(0, 256))
    #     labels_ture = np.empty(shape=(0,))
    #     model.eval()
    #     with torch.no_grad():
    #         for x, w, y in Loader_all:
    #             hiddens = np.concatenate(
    #                 [hiddens, model.ucd(x[:, 0, :].cuda(), x[:, 1, :].cuda(), w.cuda(), w.cuda()).cpu().numpy()])
    #             labels_ture = np.concatenate([labels_ture, y.cpu().numpy()])
    # np.save(f'{dataset_name}_features.npy', hiddens)
    # np.save(f'{dataset_name}_labels.npy', labels_ture)
    # print('save done')
    dataset_name = 'BayArea'
    hiddens = np.load(f'{dataset_name}_features.npy')
    labels_ture = np.load(f'{dataset_name}_labels.npy')
    from sklearn.preprocessing import minmax_scale,normalize
    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE

    hiddens = minmax_scale(hiddens, axis=1)

    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # X_embedded = tsne.fit_transform(hiddens)
    #
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_ture, cmap='jet', alpha=0.7)
    # plt.colorbar()
    # plt.title("t-SNE Visualization of Digits Dataset")
    # plt.show()

    kmeans = KMeans(n_clusters=2,)
    kmeans.fit(hiddens)

    # 获取聚类中心
    # centroids = kmeans.cluster_centers_
    # 获取每个样本的类别

    labels = kmeans.labels_ + 1
    labels = labels.flatten()

    ari_score = adjusted_rand_score(labels_ture, labels)
    nmi_score = normalized_mutual_info_score(labels_ture, labels)
    print("Adjusted Rand Index (ARI):", ari_score)
    print("Normalized Mutual Information (NMI):", nmi_score)
    oa, best_pred_labels = cluster_accuracy(labels_ture, labels)

    # 计算 AA（平均类别准确率）和 Kappa
    aa, kappa = compute_aa_kappa(labels_ture, best_pred_labels)

    CM = confusion_matrix(labels_ture, best_pred_labels)

    Pr, Re, F1 = calculate_F1PrRe(CM)

    print(f"Overall Accuracy (OA): {oa:.4f}")
    print(f"Average Accuracy (AA): {aa:.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print(f"F1: {F1:.4f}")
    print(f"Precision: {Pr:.4f}")
    print(f"Recall: {Re:.4f}")

    a = 0


