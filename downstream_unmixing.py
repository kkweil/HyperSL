from colorama import Fore
import warnings
from scipy import io as sio

warnings.filterwarnings("ignore")
import pickle
from collections import OrderedDict
from engine.unmixing import UnmixingModel
from engine.VCA import vca
from torch.utils.data import Dataset, DataLoader, Sampler
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import spectral
import random
import torch.utils.data as Data
import torchvision.transforms as transform
from mmengine.optim import build_optim_wrapper


class HyperDataset(Dataset):
    def __init__(self, data, wave, transform=None):
        self.data = data
        self.wave = np.asarray(wave).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx]).permute(1, 2, 0), self.wave
        else:
            return self.data[idx].permute(1, 2, 0), self.wave


def read_Samson(hdr_path, img_path, gt_path):
    datas = spectral.envi.open(hdr_path)
    img = loadmat(img_path)['V'] / datas.scale_factor
    wavelength = [float(i) for i in datas.metadata['wavelength']]
    GT = loadmat(gt_path)
    abundance = GT['A']
    n_cols, n_rows, n_bands = datas.ncols, datas.nrows, datas.nbands
    abundance = np.reshape(abundance, [abundance.shape[0], n_rows, n_cols])
    end_mem = GT['M'].transpose([1, 0])
    ref_end_mem, _, _ = vca(img, len(end_mem))
    img = np.reshape(img, [n_bands, n_rows, n_cols])
    return img, wavelength, abundance, end_mem, ref_end_mem


def read_JasperRidge(hdr_path, img_path, gt_path):
    datas = spectral.envi.open(hdr_path)
    img = loadmat(img_path)['Y'] / datas.scale_factor

    wavelength = [float(i) for i in datas.metadata['wavelength']]
    GT = loadmat(gt_path)
    abundance = GT['A']
    n_cols, n_rows, n_bands = datas.ncols, datas.nrows, datas.nbands
    abundance = np.reshape(abundance, [abundance.shape[0], n_rows, n_cols])
    end_mem = GT['M'].transpose([1, 0])
    ref_end_mem, _, _ = vca(img, len(end_mem))
    img = np.reshape(img, [n_bands, n_rows, n_cols]).transpose([1, 2, 0])
    return img, wavelength, abundance, end_mem, ref_end_mem


def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    #中心区域
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


def get_train_data(mirror_image, band, train_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("**************************************************")
    return x_train.squeeze()


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def get_train_patch_seed_i(img, wavelength, img_size, train_number, batch_size, seed_i):
    seed = seed_i
    np.random.seed(seed)
    random.seed(seed)

    height, width, band = img.shape
    data_label = np.ones([height, width])
    position = np.array(np.where(data_label == 1)).transpose(1, 0)
    selected_i = np.random.choice(position.shape[0], int(train_number), replace=False)
    selected_i = position[selected_i]

    TR = np.zeros(data_label.shape)
    for i in range(int(train_number)):
        TR[selected_i[i][0], selected_i[i][1]] = 1
    total_pos_train = np.argwhere(TR == 1)
    mirror_img = mirror_hsi(height, width, band, img, patch=img_size)
    img_train = get_train_data(mirror_img, band, total_pos_train,
                               patch=img_size, band_patch=1)
    # load data
    train_transform = transform.Compose([
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomVerticalFlip(0.5)])
    img_train = torch.from_numpy(img_train).type(torch.FloatTensor)
    Label_train = HyperDataset(img_train.permute(0, -1, 1, 2), wavelength, train_transform)

    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)

    return label_train_loader


def whole2_train_and_test_data(img_size, img):
    H0, W0, C = img.shape
    if H0 < img_size:
        gap = img_size - H0
        mirror_img = img[(H0 - gap):H0, :, :]
        img = np.concatenate([img, mirror_img], axis=0)
    if W0 < img_size:
        gap = img_size - W0
        mirror_img = img[:, (W0 - gap):W0, :]
        img = np.concatenate([img, mirror_img], axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H + 1) * img_size - H
        mirror_img = img[(H - gap):H, :, :]
        img = np.concatenate([img, mirror_img], axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap):W, :]
        img = np.concatenate([img, mirror_img], axis=1)
        # gap = img_size - num_W*img_size
        # img = img[:,(W - gap):W,:]
    H, W, C = img.shape
    print('padding img:', img.shape)

    num_H = H // img_size
    num_W = W // img_size

    sub_imgs = []
    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :]
            sub_imgs.append(z)
    sub_imgs = np.array(sub_imgs)
    return sub_imgs, num_H, num_W


def SAD_loss(y_true, y_pred):
    y_true = torch.nn.functional.normalize(y_true, dim=1, p=2)
    y_pred = torch.nn.functional.normalize(y_pred, dim=1, p=2)

    A = torch.mul(y_true, y_pred)
    A = torch.sum(A, dim=1)
    sad = torch.acos(A)
    loss = torch.mean(sad)
    return loss


def numpy_SAD(y_true, y_pred):
    return np.arccos(y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    dict = {}
    SAD_ = []
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if minimum == 100:
            break
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        dict[index[1]] = index[0]  # keep Gt at first,
        SAD_.append(minimum)
        # dict[index[0]] = index[1]
        sad_mat[index[0], index[1]] = 100
        rows += 1
        sad_mat[index[0], :] = 100
        sad_mat[:, index[1]] = 100
    SAD_ = np.array(SAD_)
    Average_SAM = np.sum(SAD_) / len(SAD_)
    return dict, SAD_, Average_SAM


def plotEndmembersAndGT(endmembers, endmembersGT, abundances):
    # endmembers & endmembersGT:[num_em, band]
    # Abundance: [num_em, H,W]
    print('predict_em:', endmembers.shape)
    print('GT_em:', endmembersGT.shape)
    num_endmembers = endmembers.shape[0]
    n = num_endmembers // 2  # how many digits we will display
    if num_endmembers % 2 != 0: n = n + 1
    # dict, sad = order_endmembers(endmembers, endmembersGT)
    dict, SAD_, Average_SAM = order_endmembers(endmembers, endmembersGT)
    SAD_ordered = []
    endmember_sordered = []
    abundance_sordered = []
    # fig = plt.figure()
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "aSAM score for all endmembers: " + format(Average_SAM, '.3f') + " radians"
    st = plt.suptitle(title)
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        endmember_sordered.append(endmembers[dict[i]])
        abundance_sordered.append(abundances[dict[i], :, :])
    endmember_sordered = np.array(endmember_sordered)
    for i in range(num_endmembers):
        z = numpy_SAD(endmember_sordered[i], endmembersGT[i, :])
        SAD_ordered.append(z)

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembersGT[i, :], 'r', linewidth=1.0, label='GT')
        plt.plot(endmember_sordered[i, :], 'k-', linewidth=1.0, label='predict')
        ax.set_title("SAD: " + str(i) + " :" + format(SAD_ordered[i], '.4f'))
        ax.get_xaxis().set_visible(False)

    SAD_ordered.append(Average_SAM)
    SAD_ordered = np.array(SAD_ordered)

    abundance_sordered = np.array(abundance_sordered)  # [3, 95, 95]

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)
    return SAD_ordered, endmember_sordered, abundance_sordered


def alter_MSE(y_true, y_pred):
    # y_true:[num_em,H,W]
    # y_pred:[num_em,H,W]
    # MSE--相当于y-y_hat的二阶范数的平方/n

    num_em = y_true.shape[0]
    y_true = np.reshape(y_true, [num_em, -1])
    y_pred = np.reshape(y_pred, [num_em, -1])

    R = y_pred - y_true
    r = R * R
    mse = np.mean(r, axis=1)
    Average_mse = np.sum(mse) / len(mse)
    mse = np.insert(mse, num_em, Average_mse, axis=0)
    return mse


def read_Unban(data_path, wave_path):
    dataset = sio.loadmat(data_path)
    data, GT, abundance = dataset['img_3d'], dataset['endmember'], dataset['abundance']
    init_em = dataset['init_em']
    waves = []
    with open(wave_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            waves.append(float(line.strip().split('\t')[1]))
    waves = np.array(waves)
    GT = GT.transpose([1, 0])
    return data, waves, abundance, GT, init_em


if __name__ == '__main__':

    W_AB, W_TV = 0.35, 0.1
    img_size = 5
    data_path = 'unmixing_data/urban/Urban_188_em4_init.mat'
    wave_path = 'unmixing_data/urban/URBAN.wvl'
    img, wavelength, abundance, end_mem, ref_end_mem = read_Unban('unmixing_data/urban/Urban_188_em4_init.mat', 'unmixing_data/urban/URBAN.wvl')
    img = img.transpose([1, 2, 0])
    # hdr_path, gt_path = 'unmixing_data/Samson/samson_1.img.hdr', 'unmixing_data/Samson/end3.mat'
    # img_path = 'unmixing_data/Samson/samson_1.mat'
    # img, wavelength, abundance, end_mem, ref_end_mem = read_Samson(hdr_path, img_path, gt_path)
    # img = img.transpose([1, 2, 0])
    # hdr_path, gt_path = 'unmixing_data/JasperRidge/jasperRidge2_R198.hdr', 'unmixing_data/JasperRidge/end4.mat'
    # img_path = 'unmixing_data/JasperRidge/jasperRidge2_R198.mat'
    # img, wavelength, abundance, end_mem, ref_end_mem = read_JasperRidge(hdr_path, img_path, gt_path)

    # loader data
    label_train_loader = get_train_patch_seed_i(img, wavelength, img_size, 400, 16, 886)

    # loader encoder
    model = UnmixingModel(len(end_mem), img.shape[-1], 'small')

    ckpt = torch.load('10_base_mask95_checkpoint.pt')
    weights = OrderedDict()
    for k, v in ckpt['model'].items():
        name = k[7:]
        weights[name] = v
    model.spectral_encoder.load_state_dict(weights)

    # loader ref_end_mem
    model_dict = model.state_dict()
    print('---------NOTE: Endmember initialized by VCA---------')
    model_dict['decoder.decoder.weight'][:, :, 0, 0] = torch.from_numpy(
        ref_end_mem).float()
    model.load_state_dict(model_dict)
    # train
    model.cuda()
    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05),
    )
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, 200, eta_min=0, last_epoch=-1)

    print("start training")
    # t0 = time.time()
    L, L_sad, L_ab, L_tv = [], [], [], []
    for _epoch in range(0, 1):
        scheduler.step()
        model.train()
        l, l_sad, l_ab, l_tv = 0, 0, 0, 0
        for i, data in enumerate(label_train_loader):
            img_cube = data[0].cuda()
            waves = data[1].cuda()
            abunds, output = model(img_cube, waves)
            loss_sad = SAD_loss(img_cube.permute(0, -1, 1, 2), output)
            loss_ab = W_AB * torch.sqrt(abunds).mean()
            endmembers = model.getEndmembers()
            loss_tv = W_TV * (torch.abs(endmembers[:, 1:] - endmembers[:, :(-1)]).sum())
            loss = loss_sad + loss_ab + loss_tv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l += loss.item()
            l_sad += loss_sad.item()
            l_ab += loss_ab.item()
            l_tv += loss_tv.item()
        l = l / (i + 1)
        l_sad = l_sad / (i + 1)
        l_ab = l_ab / (i + 1)
        l_tv = l_tv / (i + 1)
        L.append(l)
        L_sad.append(l_sad)
        L_ab.append(l_ab)
        L_tv.append(l_tv)
        if _epoch % 1 == 0:
            print('epoch [{}/{}],train loss:{:.4f},loss_sad:{:.4f},loss_ab:{:.4f},loss_tv:{:.4f}'
                  .format(_epoch + 1, 200, l, l_sad, l_ab, l_tv))
    # print('---training is successfully down---')
    model_save_PATH = f'unmixing_jasperRidge.pkl'
    print('Model PATH:', model_save_PATH)
    with open(model_save_PATH, 'wb') as f:
        pickle.dump(model, f)
    # test
    x_test, num_H, num_W = whole2_train_and_test_data(img_size, img)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    Label_test = HyperDataset(x_test.permute(0, -1, 1, 2), wavelength)
    label_test_loader = Data.DataLoader(Label_test, batch_size=num_H, shuffle=False)
    with open(model_save_PATH, 'rb') as f:
        model = pickle.load(f)
    model.eval()
    endmembers = model.getEndmembers().detach().cpu().numpy()
    num_em = len(end_mem)
    pred = torch.zeros([num_W, num_H, num_em, img_size, img_size])  # for test_loader, batch_size=num_H
    for batch_idx, (batch_data) in enumerate(label_test_loader):
        img_cubes = batch_data[0].cuda()
        waves = batch_data[1].cuda()
        batch_pred = model.getAbundances(img_cubes, waves).detach().cpu()
        pred[batch_idx] = batch_pred
    pred = torch.permute(pred, [2, 0, 3, 1, 4])
    abundances = np.reshape(pred, [num_em, num_H * img_size, num_W * img_size])
    abundances = abundances[:, :img.shape[0], :img.shape[1]]
    abundances = np.array(abundances)
    print('Abundance: ', abundances.shape)
    SAD_ordered, endmember_sordered, abundance_sordered = plotEndmembersAndGT(endmembers.T, end_mem,
                                                                              abundances)

    mse = alter_MSE(abundance, abundance_sordered)  # y_true, y_pred
    print("SAD_repeat_i:[SAD_i, avg_SAD]", '\n', SAD_ordered)
    print("MSE_repeat_i:[mse_i, avg_mse]", '\n', mse)

    print('Training is down!')
    result_save_file = f'unmixing_jasperRidge_result.mat'
    print('save_file:', result_save_file)
    sio.savemat(result_save_file,
                {'decoder_weight': endmembers, 'endmembers': endmember_sordered,
                 'abundances': abundance_sordered, 'SAD': SAD_ordered, 'MSE': mse})
    a = 0
