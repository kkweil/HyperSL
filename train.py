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


def scheduler_lr(optimizer, scheduler):
    lr_history = []

    """optimizer的更新在scheduler更新的前面"""
    for epoch in range(2):
        for i in range(32000):
            optimizer.step()  # 更新参数
            lr_history.append(optimizer.param_groups[0]['lr'])
            scheduler.step()  # 调整学习率
    return lr_history


if __name__ == '__main__':
    EPOCH = 10
    start_epoch = 0
    lr = 5e-4
    batch_size = 256
    mask_ratio = 0.95
    resume = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MHSIs = MultisourceHSI(dest='./MultiSourceHSI.hdf5')
    EnMap_dataset = MultisourceHSIDataset(MHSIs, source='ENMap')
    DESIS_dataset = MultisourceHSIDataset(MHSIs, source='DESIS')
    EnMap_dataloader = DataLoader(EnMap_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    DESIS_dataloader = DataLoader(DESIS_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    model = SpectralSharedEncoder(embedding_dim=256,
                                  num_heads=8,
                                  decoder_depth=4,
                                  encoder_depth=8,
                                  )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=5e-5,
        eps=1e-8,
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, )

    writer = SummaryWriter()

    if resume:
        checkpoint = torch.load('datautils')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # load checkpoint

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    length = min(len(EnMap_dataloader), len(DESIS_dataloader))
    step = 0
    for epoch in range(start_epoch, EPOCH):

        model.train()
        print('epoch', epoch, ':')
        for (enmap_s, enmap_w), (desis_s, desis_w) in zip(EnMap_dataloader, DESIS_dataloader):
            step += 1
            _, output1 = model(enmap_s.cuda(), enmap_w.cuda(), mask_ratio)
            _, output2 = model(desis_s.cuda(), desis_w.cuda(), mask_ratio)
            loss1 = MSE_SAM_loss(output1, enmap_s.cuda())
            loss2 = MSE_SAM_loss(output2, desis_s.cuda())
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            scheduler.step()

            print(f'Step:{step}', ': total_loss:', loss.item(), 'loss1:', loss1.item(), 'loss2:', loss2.item(),
                  'lr:', scheduler.get_last_lr())

            writer.add_scalar('Total_loss/train', loss.item(), step)
            writer.add_scalar('Loss1/train', loss1.item(), step)
            writer.add_scalar('Loss2/train', loss2.item(), step)

            if step % 5000 == 0:
                plt.plot(output1.detach().cpu().numpy().squeeze()[10, :], label='en_r')
                plt.plot(enmap_s.detach().cpu().numpy().squeeze()[10, :], label=f'en_o{loss1.item()}')
                plt.plot(output2.detach().cpu().numpy().squeeze()[10, :], label='de_r')
                plt.plot(desis_s.detach().cpu().numpy().squeeze()[10, :], label=f'de_o{loss2.item()}')
                plt.legend()
                plt.ylim((0, 1.0))
                plt.show()

        with torch.no_grad():
            pass
            # test

        # save model
        if epoch % 5 == 0:
            # save checkpoint
            to_save = {
                'epoch':epoch,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
            }
            path_checkpoint = f'modelarchive/{epoch}_checkpoint.pt'
            torch.save(to_save,path_checkpoint)



    a = 0
