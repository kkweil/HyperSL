import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from datautils.mutihsi import MultisourceHSI, MultisourceHSIDataset
from engine.model import SpectralSharedEncoder
from engine.loss import MSE_SAM_loss


if __name__ == '__main__':
    EPOCH = 10
    lr = 1e-3
    batch_size = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MHSIs = MultisourceHSI(dest='./MultiSourceHSI.hdf5')
    EnMap_dataset = MultisourceHSIDataset(MHSIs, source='ENMap')
    DESIS_dataset = MultisourceHSIDataset(MHSIs, source='DESIS')
    EnMap_dataloader = DataLoader(EnMap_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    DESIS_dataloader = DataLoader(DESIS_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    model = SpectralSharedEncoder(embedding_dim=256,
                                  num_heads=16,
                                  decoder_depth=8,
                                  encoder_depth=8,
                                  )

    dict1 = torch.load('model_dict_step2.pth')
    dict2 = torch.load('model_dict_step3.pth')


    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=1e-3,
        )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-8)

    for (enmap_s, enmap_w), (desis_s, desis_w) in zip(EnMap_dataloader, DESIS_dataloader):
        _, output1 = model(enmap_s.cuda(), enmap_w.cuda())
        _, output2 = model(desis_s.cuda(), desis_w.cuda())

    a = 0


