import numpy as np
from DataUtilization import get_dataarchive, ReflectionPaddding
from engine.model import AE1DCLF
from torch.utils.data import DataLoader
import math
# from skimage.metrics import mean_squared_error as compare_mse
import torch.utils.data
from sklearn.preprocessing import minmax_scale
from colorama import Fore, init

init()


def preprocess(data):
    shape = data.shape  # w,h,c
    # shapes.append(shape)
    data = data.reshape(-1, shape[-1])

    data = minmax_scale(data).astype('float32')
    data = data.reshape(shape[0], shape[1], -1)

    return data


def get_data(train_ratio, data, gt):
    data = preprocess(data)
    coordinates = {i: np.argwhere(gt == i + 1) for i in range(0, gt.max())}
    for k, v in coordinates.items():
        np.random.shuffle(v)
    coordinate_train = [v[:math.ceil(len(v) * train_ratio)].tolist() for k, v in coordinates.items()]
    coordinate_train = np.asarray([i for list_ in coordinate_train for i in list_])

    coordinate_test = [v[math.ceil(len(v) * train_ratio):].tolist() for k, v in coordinates.items()]
    coordinate_test = np.asarray([i for list_ in coordinate_test for i in list_])
    x_train = np.asarray([data[coordinate_train[i][0], coordinate_train[i][1]]
                          for i in range(len(coordinate_train))])
    x_test = np.asarray([data[coordinate_test[i][0], coordinate_test[i][1]]
                         for i in range(len(coordinate_test))])
    y_train = np.asarray([gt[coordinate_train[i][0], coordinate_train[i][1]]
                          for i in range(len(coordinate_train))]).astype('int64') - 1
    y_test = np.asarray([gt[coordinate_test[i][0], coordinate_test[i][1]]
                         for i in range(len(coordinate_test))]).astype('int64') - 1

    return np.expand_dims(x_train, axis=1), y_train, np.expand_dims(x_test, axis=1), y_test


class ClfDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, shape, Cr):
        self.data = data
        self.shape = shape
        self.label = label
        self.Cr = Cr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return ReflectionPaddding(self.data[idx], bands=self.shape[-1], Cr=self.Cr), self.label[idx]


if __name__ == '__main__':
    Cr = 32
    epochs = 200
    learning_rate = 0.001
    pertrain = True
    testdata_archive, _ = get_dataarchive('../testdata')
    # img = testdata_archive['clfdata']['indian_pines_corrected_200']
    # gt = testdata_archive['gt']['indian_pines_gt']
    img = testdata_archive['clfdata']['paviaU_103']
    gt = testdata_archive['gt']['paviaU_gt']

    # testdata = get_spectra(**testdata_archive)
    x_train, y_train, x_test, y_test = get_data(0.01, img, gt)
    train_set = ClfDataset(x_train, y_train, img.shape, Cr)
    test_set = ClfDataset(x_test, y_test, img.shape, Cr)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_Loader = DataLoader(test_set, batch_size=4000, shuffle=True)

    model = AE1DCLF(cls_num=9, latten_dim=128, embedding_dim=512, hidden_dim=1)

    # load masked autoencoder pretrain
    if pertrain:
        pretrain_dict = torch.load(r'runs/exp1/best.pth')
        clf_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if k in clf_dict}
        clf_dict.update(pretrain_dict)
        model.load_state_dict(clf_dict)

    if torch.cuda.is_available():
        model.cuda()

    optimzier = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss()
    flag = 0

    for epoch in range(epochs):
        model.train()
        loss_ = []
        correct = 0
        for x, y in train_loader:
            y_pred = model(x.cuda())
            loss = loss_fn(y_pred, y.cuda())
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()
            loss_.append(loss.item())
            correct = correct + (y_pred.argmax(1) == y.cuda()).sum().item()

        print(Fore.RESET+f'Epoch{epoch+1}: mean_loss:{sum(loss_) / len(loss_)} acc:{correct / len(x_train)}')
        flag += 1
        if 0 == flag % 10:
            pred = np.empty(shape=(0,))
            labels = np.empty(shape=(0,))
            with torch.no_grad():
                for x, y in test_Loader:
                    pred = np.concatenate([pred, model(x.cuda()).argmax(1).cpu().numpy()])
                    labels = np.concatenate([labels, y.cpu().numpy()])
                    # loss = loss_fn(y_pred, y.cuda())
                    # loss_.append(loss.item())
                    # correct = correct + (y_pred.argmax(1) == y.cuda()).sum().item()
                pred = pred.astype('int')
                labels = labels.astype('int')
                cls_nums = {}
                for i in range(labels.max()):
                    cls_nums[i] = sum(labels == i)
                from sklearn.metrics import confusion_matrix

                CM = confusion_matrix(labels, pred)
                aa = []
                for i in range(labels.max()):
                    aa.append(CM[i, i] / cls_nums[i])
                AA = sum(aa) / len(aa)

                l = pred - labels
                OA = sum(l == 0) / len(l)

                k = 0
                for i in range(labels.max()):
                    c = sum(CM[i, :])
                    r = sum(CM[:, i])
                    k = k + r * c
                k = k / (len(pred) ** 2)
                Kappa = (OA - k) / (1 - k)
                print(Fore.RED + f'Epoch{epoch+1}: OA:{OA} AA:{AA} Kappa:{Kappa}')
