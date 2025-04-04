import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import DropPath


class SpectralSharedEncoder(nn.Module):
    def __init__(self,
                 input_dim=1024,
                 output_dim=1024,
                 hidden_dim=1024,
                 ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.encoder(x)
        x = self.decoder(h)
        return x, h


if __name__ == '__main__':

    # input1 降采样 5*5

    input1 = torch.randn(16, 4)
    input2 = torch.randn(16, 300)

    model = SpectralSharedEncoder(
        input_dim=304,
        hidden_dim=32,
        output_dim=300,
    )

    model.cuda()

    x, h = model(input1.cuda(), input2.cuda())
    a = 0
