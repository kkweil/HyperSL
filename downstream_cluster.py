import spectral
import numpy as np
from engine.model import SpectralSharedEncoder
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import faiss
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim

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

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SpectralDataset(Dataset):
    def __init__(self, data, wavelength):
        self.data = data
        self.wavelength = wavelength

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.wavelength


img_data = spectral.open_image(r'Hyperspec_Chikusei_ENVI/Chikusei_ENVI/HyperspecVNIR_Chikusei_20140729.hdr')
lable_data = spectral.open_image(
    r'Hyperspec_Chikusei_ENVI/Chikusei_ENVI/HyperspecVNIR_Chikusei_20140729_Ground_Truth.hdr')

img = img_data.load()
lable = lable_data.load().astype('int').squeeze()
lable_cluster = np.zeros_like(lable)

wavelength = torch.tensor([float(i) * 1000 for i in img_data.metadata['wavelength']])

args = np.where(lable != 0)

data_ori = img[args].astype('float32')


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Autoencoder().to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# dataset = SpectralDataset(data_ori, wavelength)
#
# train_loader = DataLoader(dataset, batch_size=5000, shuffle=True)
# test_loader = DataLoader(dataset, batch_size=5000, shuffle=False)
#
# num_epochs = 100
# for epoch in range(num_epochs):
#     for data, _ in train_loader:
#         data = data.to(device)
#         output = model(data)
#         loss = criterion(output, data)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
#
# hiddens = np.empty(shape=(0, 32))
# with torch.no_grad():
#     model.eval()
#     for data, _ in test_loader:
#         data = data.to(device)
#         h = model.encoder(data)
#         hiddens = np.concatenate([hiddens, h.detach().cpu().numpy().squeeze()], axis=0)
# # a = 0
# # pca = PCA(n_components=0.98)
# # data_pca = pca.fit_transform(data_ori)
# #
# np.save('hiddens_AE.npy', hiddens)
# hiddens = np.load('hiddens_AE.npy')
# kmeans = AgglomerativeClustering(n_clusters=19, )
# kmeans.fit(hiddens)
#
# # 获取聚类中心
# # centroids = kmeans.cluster_centers_
# # 获取每个样本的类别
# labels = kmeans.labels_+1
# labels = labels.flatten()
#
# ari_score = adjusted_rand_score(lable[args], labels)
# nmi_score = normalized_mutual_info_score(lable[args], labels)
# print("Adjusted Rand Index (ARI):", ari_score)
# print("Normalized Mutual Information (NMI):", nmi_score)
#
# oa, best_pred_labels = cluster_accuracy(lable[args], labels)
#
# # 计算 AA（平均类别准确率）和 Kappa
# aa, kappa = compute_aa_kappa(lable[args], best_pred_labels)
#
# print(f"Overall Accuracy (OA): {oa:.4f}")
# print(f"Average Accuracy (AA): {aa:.4f}")
# print(f"Kappa Coefficient: {kappa:.4f}")

# lable_cluster[args] = labels+1
# import matplotlib.pyplot as plt
# plt.imshow(lable_cluster)
# plt.show()

#
# traindata = torch.tensor(img[args])[:, 8:].unsqueeze(1)
#
# model = SpectralSharedEncoder(embedding_dim=256,
#                               num_heads=8,
#                               decoder_depth=4,
#                               encoder_depth=8,
#                               )
#
# ckpt = torch.load('10_base_mask95_checkpoint.pt')
# weights = OrderedDict()
# for k, v in ckpt['model'].items():
#     name = k[7:]
#     weights[name] = v
# model.load_state_dict(weights)
#
# dataset = SpectralDataset(traindata, wavelength)
# dataloader = DataLoader(dataset, batch_size=3000, shuffle=False)
#
# model.cuda()
# model.eval()
# hiddens = np.empty(shape=(0, 256))
# with torch.no_grad():
#     for data, wave in dataloader:
#         h, _ = model(data.cuda(), wave.cuda(), 0.)
#         hiddens = np.concatenate([hiddens, h.detach().cpu().numpy().squeeze()], axis=0)
#
# np.save('hiddens.npy', hiddens)
# hiddens = np.load('hiddens.npy')
from sklearn.preprocessing import minmax_scale,normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# hiddens = minmax_scale(hiddens, axis=1)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(data_ori)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=lable[args], cmap='jet', alpha=0.7)
plt.colorbar()
plt.title("Raw data t-SNE Visualization")
plt.show()

# kmeans = AgglomerativeClustering(n_clusters=19, )
# kmeans.fit(hiddens)
#
# # 获取聚类中心
# # centroids = kmeans.cluster_centers_
# # 获取每个样本的类别
# labels = kmeans.labels_+1
# labels = labels.flatten()
#
# ari_score = adjusted_rand_score(lable[args], labels)
# nmi_score = normalized_mutual_info_score(lable[args], labels)
# print("Adjusted Rand Index (ARI):", ari_score)
# print("Normalized Mutual Information (NMI):", nmi_score)
#
#
#
# # 示例数据
# oa, best_pred_labels = cluster_accuracy(lable[args], labels)
#
# # 计算 AA（平均类别准确率）和 Kappa
# aa, kappa = compute_aa_kappa(lable[args], best_pred_labels)
#
# print(f"Overall Accuracy (OA): {oa:.4f}")
# print(f"Average Accuracy (AA): {aa:.4f}")
# print(f"Kappa Coefficient: {kappa:.4f}")

a = 0
