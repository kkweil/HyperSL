import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from collections import OrderedDict
import matplotlib.pyplot as plt
# import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datautils.mutihsi import MultisourceHSI, MultisourceHSIDataset
from engine.model import SpectralSharedEncoder
from engine.loss import MSE_SAM_loss
from torch.nn.utils import clip_grad_norm_


import numpy as np

# 计算SID
def SID(x,y):
    # print(np.sum(x))
    p = np.zeros_like(x.squeeze())
    q = np.zeros_like(y.squeeze())
    Sid = 0
    for i in range(len(x)):
        p[i] = x[i]/np.sum(x)
        q[i] = y[i]/np.sum(y)
        # print(p[i],q[i])
    for j in range(len(x)):
        Sid += p[j]*np.log10(p[j]/(q[j]+1e-8))+q[j]*np.log10(q[j]/(p[j]+1e-8))
    return Sid

# 计算SAM
def SAM(x,y):
    s = np.sum(np.multiply(x.squeeze(),y.squeeze()),axis=1)
    t = np.sqrt(np.sum(x.squeeze()**2,axis=1))*np.sqrt(np.sum(y.squeeze()**2,axis=1))
    th = np.arccos(s/t).mean()
    th = np.rad2deg(th)
    # print(s,t)
    return th


if __name__ == '__main__':
    EPOCH = 10
    start_epoch = 0
    lr = 5e-4
    batch_size = 4000
    mask_ratio = 0.95
    resume = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MHSIs = MultisourceHSI(dest='./MultiSourceHSI_test.hdf5')
    EnMap_dataset = MultisourceHSIDataset(MHSIs, source='ENMap')
    DESIS_dataset = MultisourceHSIDataset(MHSIs, source='DESIS')
    EnMap_dataloader = DataLoader(EnMap_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    DESIS_dataloader = DataLoader(DESIS_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    model = SpectralSharedEncoder(embedding_dim=256,
                                  num_heads=8,
                                  decoder_depth=4,
                                  encoder_depth=8,
                                  )


    model = torch.nn.DataParallel(model)
    # load checkpoint
    ckpt = torch.load('10_base_mask95_checkpoint.pt')
    # weights = OrderedDict()
    # for k, v in ckpt['model'].items():
    #     name = k[7:]
    #     weights[name] = v
    model.load_state_dict(ckpt['model'])


    model.cuda()
    length = min(len(EnMap_dataloader), len(DESIS_dataloader))

    L = []
    sam_list = []
    with torch.no_grad():
        for (enmap_s, enmap_w)in DESIS_dataloader:
            _, output1 = model(enmap_s.cuda(), enmap_w.cuda(), 0.95)
            loss = MSE_SAM_loss(output1, enmap_s.cuda())
            # sid = SID(output1.cpu().numpy(),enmap_s.cpu().numpy())
            sam = SAM(output1.cpu().numpy(),enmap_s.cpu().numpy())
            print(loss.item(),sam)
            L.append(loss.item())
            sam_list.append(sam)



    print('loss:',sum(L)/len(L),'sam:',sum(sam_list)/len(sam_list))

    a = 0
