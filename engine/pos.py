import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from datautils.readmetadata import readcenterwavelength
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 初始化dropout层

        # 计算位置编码并将其存储在pe张量中
        pe = torch.zeros(max_len, d_model)  # 创建一个max_len x d_model的全零张量
        position = torch.arange(0, max_len).unsqueeze(1)  # 生成0到max_len-1的整数序列，并添加一个维度
        # 计算div_term，用于缩放不同位置的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        # 使用正弦和余弦函数生成位置编码，对于d_model的偶数索引，使用正弦函数；对于奇数索引，使用余弦函数。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 在第一个维度添加一个维度，以便进行批处理
        self.register_buffer('pe', pe)  # 将位置编码张量注册为缓冲区，以便在不同设备之间传输模型时保持其状态

    # 定义前向传播函数
    def forward(self, x):
        return self.pe


class WavePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(WavePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(100.0) / d_model))

    def forward(self, pos):
        pos = torch.tensor((pos-400)/10).unsqueeze(-1)
        pe = torch.zeros(pos.shape[0], self.d_model)
        pe[:pos.shape[0], 0::2] = torch.sin(self.div_term * pos)
        pe[:pos.shape[0], 1::2] = torch.cos(self.div_term * pos)
        return pe


if __name__ == '__main__':
    wave = np.array(readcenterwavelength('ENMAP01_METADATA.XML')).astype('float')
    # wave = np.arange(400,2500,2.5)
    pos_embedding = WavePositionalEncoding(d_model=128, max_len=1000)
    # pos_embedding1 = PositionalEncoding(d_model=128, max_len=1000)

    p = pos_embedding(wave).numpy()
    # p1 = pos_embedding1(0).numpy()
    # viridis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    im = ax1.imshow(p[:wave.shape[0], :], cmap='plasma', aspect='auto')
    ax1.set_xlabel('Embedding Dimensions')
    ax1.set_ylabel('Position in Sequence')
    ax1.plot(((wave-np.min(wave)) / (np.max(wave)-np.min(wave)) * 127), range(wave.shape[0]), color='lime', linewidth=3, label='Wavelength')
    ax1.set_xlim((0, 127))
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    tick_pos = np.linspace(0, wave.shape[0]-1, 10, endpoint=True).astype('int')
    tick_labels = [f'{int(wave[i])}' for i in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Wavelength(nm)')

    ax1.legend(loc='upper right')
    plt.colorbar(im, ax=ax1)
    plt.show()
    a = 0
