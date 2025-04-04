import os
import random
from abc import ABC

import numpy as np
from scipy.io import loadmat
import torch
from sklearn import preprocessing
# import spectral
# import wx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler
from random import sample, shuffle
import math


class HSIloader:
    def __init__(self, img_path, gt_path, patch_size, sample_mode=None, train_ratio=None,
                 sample_points=None, shots=None, query_nums=None, merge=None, rmbg=True):
        """

        :param img_path:
        :param gt_path:
        :param patch_size:
        :param sample_mode:
        :param sample_ratio:
        :param sample_points:
        :param merge:
        :param rmbg:
        """
        self.gt = loadmat(gt_path)
        self.img = loadmat(img_path)
        self.patch_size = patch_size
        self.sample_mode = sample_mode
        self.train_ratio = train_ratio
        self.sample_points = sample_points
        self.shots = shots
        self.query_nums = query_nums
        img_name = [t for t in list(self.img.keys()) if not t.startswith('__')][0]
        gt_name = [t for t in list(self.gt.keys()) if not t.startswith('__')][0]
        self.img = self.img[img_name]
        self.gt = self.gt[gt_name]
        self.merge = merge
        self.rmbg = rmbg
        self.coordinate_test = None
        self.coordinate_train = None
        self.shape = self.img.shape
        self.x_train_patch = None
        self.x_test_patch = None
        self.x_train_spectral = None
        self.x_test_spectral = None

    # def show(self):
    #     app = wx.App()
    #     spectral.settings.WX_GL_DEPTH_SIZE = 16
    #     spectral.view_cube(self.img, bands=[29, 19, 9])
    #     app.MainLoop()

    def relabel(self):
        temp = list(set(self.gt.reshape(-1).tolist()))
        temp.sort()
        for index, item in enumerate(temp):
            self.gt[self.gt == item] = index

    def preprosses(self):
        # ---------------min-max---------------
        # self.img = self.img.transpose((2, 0, 1))
        # self.img = np.asarray([(i - i.min()) / (i.max() - i.min()) for i in self.img])
        # ----------------normal---------------
        shape = self.img.shape
        self.img = np.reshape(self.img, (-1, self.img.shape[2]))
        self.img = preprocessing.scale(self.img)
        self.img = np.reshape(self.img, (shape[0], shape[1], shape[2]))
        self.img = self.img.transpose((2, 0, 1))
        if self.merge:
            for i in self.merge:
                self.gt[self.gt == i] = 0
            print(f"Merged classes {self.merge}")
            self.relabel()
        if self.rmbg:
            flt1 = np.where(self.gt >= 1, 1, 0)
            self.img = flt1 * self.img
            print("Removed background.")
        # img = img.transpose((1, 2, 0))

    def __call__(self, *args, **kwargs):
        if self.sample_mode == 'ratio':
            print(f"Sample the training dataset at a scale of {self.train_ratio}")
            assert isinstance(self.train_ratio, float), 'sample ratio must be float.'
            assert self.patch_size % 2 == 1, 'The window size must be odd.'
            R = int(self.patch_size / 2)
            self.preprosses()
            self.gt = gt = np.pad(self.gt, R, mode='constant')
            hsi = np.asarray([(np.pad(i, R, mode='constant')) for i in self.img])
            # self.img = self.img.transpose((1, 2, 0))

            coordinate = {i: np.argwhere(gt == i + 1) for i in range(0, gt.max())}
            for _, v in coordinate.items():
                np.random.shuffle(v)

            self.coordinate_train = [v[:math.ceil(len(v) * self.train_ratio)].tolist() for k, v in coordinate.items()]
            self.coordinate_train = np.asarray([i for list_ in self.coordinate_train for i in list_])

            self.coordinate_test = [v[math.ceil(len(v) * self.train_ratio):].tolist() for k, v in coordinate.items()]
            self.coordinate_test = np.asarray([i for list_ in self.coordinate_test for i in list_])

            self.x_train_patch = np.asarray([hsi[:, self.coordinate_train[i][0] - R:self.coordinate_train[i][0] + R + 1,
                                             self.coordinate_train[i][1] - R:self.coordinate_train[i][1] + R + 1]
                                             # .transpose(1, 2, 0)
                                             for i in range(len(self.coordinate_train))])
            self.x_test_patch = np.asarray([hsi[:, self.coordinate_test[i][0] - R:self.coordinate_test[i][0] + R + 1,
                                            self.coordinate_test[i][1] - R:self.coordinate_test[i][1] + R + 1]
                                            # .transpose(1, 2, 0)
                                            for i in range(len(self.coordinate_test))])
            # if not kwargs['spectral']:
            #     return self.x_train_patch, self.x_test_patch
            if kwargs['spectral']:
                self.x_train_spectral = np.asarray([hsi[:, self.coordinate_train[i][0], self.coordinate_train[i][1]]
                                                    for i in range(len(self.coordinate_train))])
                self.x_test_spectral = np.asarray([hsi[:, self.coordinate_test[i][0], self.coordinate_test[i][1]]
                                                   for i in range(len(self.coordinate_test))])
                # return self.x_train_patch, self.x_test_patch, self.x_train_spectral, self.x_test_spectral

        elif self.sample_mode == 'counts':
            print(f"Sample the training dataset  {self.sample_points} points per class.")
            assert isinstance(self.sample_points, int), 'sample points must be int.'
            R = int(self.patch_size / 2)
            self.preprosses()
            self.gt = gt = np.pad(self.gt, R, mode='constant')
            hsi = np.asarray([(np.pad(i, R, mode='constant')) for i in self.img])
            coordinate = {i: np.argwhere(gt == i + 1) for i in range(0, gt.max())}
            for _, v in coordinate.items():
                np.random.shuffle(v)

            self.coordinate_train = []
            self.coordinate_test = []
            for k, v in coordinate.items():
                if len(v) > 2 * self.sample_points:
                    self.coordinate_train.append(v[:self.sample_points].tolist())
                    self.coordinate_test.append(v[self.sample_points:].tolist())
                else:
                    self.coordinate_train.append(v[:int(len(v) / 2)].tolist())
                    self.coordinate_test.append(v[int(len(v) / 2):].tolist())

            # self.coordinate_train = [v[:self.sample_points].tolist() for k, v in coordinate.items()]
            # self.coordinate_test = [v[self.sample_points:].tolist() for k, v in coordinate.items()]
            self.coordinate_train = np.asarray([i for list_ in self.coordinate_train for i in list_])
            self.coordinate_test = np.asarray([i for list_ in self.coordinate_test for i in list_])

            self.x_train_patch = np.asarray([hsi[:, self.coordinate_train[i][0] - R:self.coordinate_train[i][0] + R + 1,
                                             self.coordinate_train[i][1] - R:self.coordinate_train[i][1] + R + 1]
                                             for i in range(len(self.coordinate_train))])
            self.x_test_patch = np.asarray([hsi[:, self.coordinate_test[i][0] - R:self.coordinate_test[i][0] + R + 1,
                                            self.coordinate_test[i][1] - R:self.coordinate_test[i][1] + R + 1]
                                            for i in range(len(self.coordinate_test))])

            if kwargs['spectral']:
                self.x_train_spectral = np.asarray([hsi[:, self.coordinate_train[i][0], self.coordinate_train[i][1]]
                                                    for i in range(len(self.coordinate_train))])
                self.x_test_spectral = np.asarray([hsi[:, self.coordinate_test[i][0], self.coordinate_test[i][1]]
                                                   for i in range(len(self.coordinate_test))])
        elif self.sample_mode == 'few-shot':
            print(f"Sample the support set  {self.shots} points per class.")
            assert isinstance(self.shots, int), 'sample points must be int.'
            R = int(self.patch_size / 2)
            self.preprosses()
            self.gt = gt = np.pad(self.gt, R, mode='constant')
            hsi = np.asarray([(np.pad(i, R, mode='constant')) for i in self.img])
            coordinate = {i: np.argwhere(gt == i + 1) for i in range(0, gt.max())}
            for _, v in coordinate.items():
                np.random.shuffle(v)
            # FIXME: some class number is less than sample points

            support_dict = {}
            query_dict = {}
            test_dict = {}
            for k, v in coordinate.items():
                if len(v) > self.shots + self.query_nums:
                    support_dict[k] = v[:self.shots].tolist()
                    query_dict[k] = v[self.shots:self.shots + self.query_nums].tolist()
                    test_dict[k] = v[self.shots + self.query_nums:].tolist()
                else:
                    support_dict[k] = v[:self.shots].tolist()
                    query_dict[k] = v[self.shots:int(len(v) / 2)].tolist()
                    test_dict[k] = v[self.shots + self.query_nums:].tolist()
            # support_dict = {k: v[:self.sample_points] for k, v in coordinate.items()}
            # query_dict = {k: v[self.sample_points:] for k, v in coordinate.items()}
            self.support_set = {k: np.asarray([hsi[:, v[i][0] - R:v[i][0] + R + 1, v[i][1] - R:v[i][1] + R + 1]
                                               for i in range(len(v))]) for k, v in support_dict.items()}
            # Todo: support_set Data Augmentation
            self.support_set_da = {}
            support_set_da_len = 200
            for k, v in self.support_set.items():
                self.support_set_da[k] = []
                for i in v:
                    self.support_set_da[k].append(i)
                for _ in range(math.ceil((support_set_da_len - self.shots) / self.shots)):
                    for i in v:
                        patch_da = i + np.random.normal(loc=0, scale=0.005, size=i.shape)
                        self.support_set_da[k].append(patch_da)

            self.query_set = {k: np.asarray([hsi[:, v[i][0] - R:v[i][0] + R + 1, v[i][1] - R:v[i][1] + R + 1]
                                             for i in range(len(v))]) for k, v in query_dict.items()}

            self.train_set = np.asarray([i for k, v in self.support_set.items() for i in v])

            self.test_set = np.asarray([hsi[:, v[i][0] - R:v[i][0] + R + 1, v[i][1] - R:v[i][1] + R + 1]
                                        for k, v in test_dict.items() for i in range(len(v))])
            self.train_set_coordinate = np.asarray([i for k, v in support_dict.items() for i in v])
            self.test_set_coordinate = np.asarray([i for k, v in test_dict.items() for i in v])

        elif self.sample_mode == 'bootstrap':
            print(f"Sample the training dataset using Bootstrap.")
            assert self.patch_size % 2 == 1, 'The window size must be odd.'
            R = int(self.patch_size / 2)
            self.preprosses()
            gt = np.pad(self.gt, R, mode='constant')
            hsi = np.asarray([(np.pad(i, R, mode='constant')) for i in self.img])
            # self.img = self.img.transpose((1, 2, 0))
            self.coordinate_test = np.argwhere(gt != 0)
            index = np.random.choice(range(len(self.coordinate_test)), size=len(self.coordinate_test), replace=True,
                                     p=None)  # uniform sample, replace=True
            self.coordinate_train = [self.coordinate_test[i] for i in index]
            self.x_train_patch = np.asarray([hsi[:, self.coordinate_train[i][0] - R:self.coordinate_train[i][0] + R + 1,
                                             self.coordinate_train[i][1] - R:self.coordinate_train[i][1] + R + 1]
                                             # .transpose(1, 2, 0)
                                             for i in range(len(self.coordinate_train))])
            self.x_test_patch = np.asarray([hsi[:, self.coordinate_test[i][0] - R:self.coordinate_test[i][0] + R + 1,
                                            self.coordinate_test[i][1] - R:self.coordinate_test[i][1] + R + 1]
                                            # .transpose(1, 2, 0)
                                            for i in range(len(self.coordinate_test))])
            # if not kwargs['spectral']:
            #     return self.x_train_patch, self.x_test_patch
            if kwargs['spectral']:
                self.x_train_spectral = np.asarray([hsi[:, self.coordinate_train[i][0], self.coordinate_train[i][1]]
                                                    for i in range(len(self.coordinate_train))])
                self.x_test_spectral = np.asarray([hsi[:, self.coordinate_test[i][0], self.coordinate_test[i][1]]
                                                   for i in range(len(self.coordinate_test))])
                # return self.x_train_patch, self.x_test_patch, self.x_train_spectral, self.x_test_spectral
        else:
            print('Please choose the sample mode from [ratio,bootstrap,counts]')


class HSIDataset(Dataset):
    def __init__(self, images, groundtruth, coordinate, transform=None):
        self.imgs = images.astype('float32')
        self.labels = np.asarray([groundtruth[coord[0], coord[1]] - 1 for coord in coordinate])
        # self.labels = np.asarray([groundtruth[coord[0], coord[1]] for coord in coordinate])
        self.coordinate = coordinate
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # image = self.imgs[idx]
        image = torch.from_numpy(self.imgs[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class GF5Dataset(Dataset):
    def __init__(self, img_root, transform=None):
        self.img_root = img_root
        self.samples = readfiles(self.img_root)
        self.transform = transform

    def readfiles(self):
        filelist = []
        for rootdir, subdirs, files in os.walk(self.img_root):
            if files and subdirs == []:
                for i in files:
                    filelist.append(os.path.join(rootdir, i))
        return filelist

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # image = self.imgs[idx]
        image = torch.from_numpy(np.load(self.samples[idx]))
        if self.transform:
            image = self.transform(image)
        return image


class FewShotDataset(Dataset):
    def __init__(self, dataset):
        super(FewShotDataset, self).__init__()
        self.data = [i[0] for i in dataset]
        self.labels = [i[1] for i in dataset]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image = self.imgs[idx]
        image = self.data[idx]
        label = self.labels[idx]
        return image, label


class ReassignedDataset(Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of clfdata indexes
        pseudolabels (list): list of labels for each clfdata
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            img = dataset[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((img, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of clfdata
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        img, pseudolabel = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''

    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True, mode=None):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        self.mode = mode

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]

        batch = [sublist[i] for i in range(self.num_per_class) for sublist in batch]
        # batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        # return 1 if self.mode == 'train' else int(self.num_inst/self.num_per_class)
        return self.num_per_class * self.num_cl


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of clfdata with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


def readfiles(root):
    filelist = []
    for rootdir, subdirs, files in os.walk(root):
        if files and subdirs == []:
            for i in files:
                filelist.append(os.path.join(rootdir, i))
    return filelist


def meta_task(support_set, query_set, ways, shots, queries, num_per_class_support, num_per_class_query, episode):
    np.random.seed(episode)
    categories = list(support_set)
    support_dataset = []
    query_dataset = []

    assert ways <= len(categories)
    class_list = random.sample(categories, ways)
    labels = list(range(ways))
    labels = dict(zip(class_list, labels))

    for c in class_list:
        temp_s = support_set[c]
        temp_q = query_set[c]
        random.shuffle(temp_s)
        random.shuffle(temp_q)
        temp_s = temp_s[:shots]
        temp_q = temp_q[:queries]
        for _ in range(shots):
            support_dataset.append((temp_s[_], labels[c]))
        for i in range(queries):
            query_dataset.append((temp_q[i], labels[c]))
    # for _ in range(shots):
    #     for c in class_list:
    #         temp_s = random.sample(support_set[c], 1)
    #         support_dataset.append((temp_s, labels[c]))
    # for _ in range(queries):
    #     for c in class_list:
    #         temp_q = random.sample(support_set[c], 1)
    #         support_dataset.append((temp_q, labels[c]))
    support_dataset = FewShotDataset(support_dataset)
    query_dataset = FewShotDataset(query_dataset)

    support_sampler = ClassBalancedSampler(num_per_class=num_per_class_support, num_cl=ways, num_inst=shots,
                                           shuffle=False, mode='train')
    query_sampler = ClassBalancedSampler(num_per_class=num_per_class_query, num_cl=ways, num_inst=queries, shuffle=True,
                                         mode='test')
    support_loader = DataLoader(support_dataset, batch_size=support_sampler.__len__(), sampler=support_sampler)
    query_loader = DataLoader(query_dataset, batch_size=18, sampler=query_sampler)

    # if ways == len(categories):
    #     for k, v in support_set.items():
    #         np.random.shuffle(v)
    #         for i in range(shots):
    #             support_dataset.append((v[i], k))
    #
    #     for k, v in query_set.items():
    #         np.random.shuffle(v)
    #         for i in range(queries):
    #             query_dataset.append((v[i], k))
    # else:
    #     diff = len(categories) - ways
    #     drop = sample(categories, diff)
    #     for i in drop:
    #         del support_set[i]
    #         del query_set[i]
    #
    #     for k, v in support_set.items():
    #         np.random.shuffle(v)
    #         for i in range(shots):
    #             support_dataset.append((v[i], k))
    #
    #     for k, v in query_set.items():
    #         np.random.shuffle(v)
    #         for i in range(queries):
    #             query_dataset.append((v[i], k))
    # shuffle(support_dataset)
    # shuffle(query_dataset)
    # if torch.cuda.is_available():
    #     pass
    return support_loader, query_loader


def spectral_padding(image, max_len: int):
    assert max_len > image.shape[1], 'Error!'
    image = image.permute(0, 2, 3, 1)  # nchw -> nhwc
    shape = image.shape
    image = image.reshape(-1, image.shape[-1])
    # image.reshape(-1, image.shape[-1])
    img_pad = np.zeros((image.shape[0], max_len))
    for idx, spectral in enumerate(image):
        length = len(spectral)
        interp = np.zeros((max_len,))
        idx_map = np.round((np.asarray(list(range(length))) + 1) * (max_len / length) - 1).astype('uint')
        for i, item in enumerate(idx_map):
            interp[item] = spectral[i]
        o_idx = np.where(interp == 0)
        for i in o_idx[0]:
            if i < (max_len - 1):
                interp[i] = (interp[i - 1] + interp[i + 1]) / 2
            else:
                interp[i] = (interp[i - 2] + interp[i - 1]) / 2
        img_pad[idx] = interp
    image_pad = torch.tensor(img_pad.reshape((shape[0], shape[1], shape[2], -1))).permute(0, 3, 1, 2)  # nhwc -> nchw
    return image_pad


if __name__ == '__main__':
    # imp = r'../clfdata/Salinas_corrected.mat'
    # gtp = r'../clfdata/Salinas_gt.mat'
    # imp = r'../clfdata/Indian_pines_corrected.mat'
    # gtp = r'../clfdata/Indian_pines_gt.mat'
    imp = r'../clfdata/PaviaU.mat'
    gtp = r'../clfdata/PaviaU_gt.mat'

    # root = r'../clfdata/GF5_patches'

    # l = readfiles(root)
    dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=31, sample_mode='few-shot', shots=5,
                        query_nums=20, merge=None, rmbg=False)
    # # dataset.show()
    SPECTRAL = False
    dataset(spectral=SPECTRAL)
    test_dataset = HSIDataset(dataset.test_set, dataset.gt, dataset.test_set_coordinate, transform=None)
    #
    # train_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    support_set_da = dataset.support_set_da
    query_set = dataset.query_set

    support_dataset, query_dataset = meta_task(support_set_da, query_set, ways=9, shots=5, queries=20,
                                               num_per_class_support=1, num_per_class_query=10, episode=10)

    a = 0

    # if not os.path.exists('./clfdata/'):
    #     os.mkdir('./clfdata')
    # with open('clfdata/Salinas.pickle', 'wb') as p:
    #     pickle.dump(dataset, p)
    # p.close()
