import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import DropPath
from .model import SpectralSharedEncoder

class ChangedetectionModel(nn.Module):
    def __init__(self, class_num , model_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_size == 'small':
            self.embedding_dim = 256
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=256,
                encoder_depth=8,
                decoder_depth=4,
                num_heads=8,
            )
        if model_size == 'base':
            self.embedding_dim = 512
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=512,
                max_band=500,
                encoder_depth=24,
                decoder_depth=12,
                num_heads=16,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )
        if model_size == 'large':
            self.embedding_dim = 1024
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=1024,
                max_band=500,
                encoder_depth=32,
                decoder_depth=16,
                num_heads=32,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )

        if model_size == 'huge':
            self.embedding_dim = 2048
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=2048,
                max_band=500,
                encoder_depth=48,
                decoder_depth=24,
                num_heads=32,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )

        self.convblock = nn.Sequential(
            nn.Conv2d(self.embedding_dim,64,1,1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 512, 3, 1), # h,w /2
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, 1),  # h,w /2
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, 3, 1),  # h,w /2
            nn.BatchNorm2d(64),
            nn.GELU(),)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,512),
            nn.Dropout(0.4),
            nn.GELU(),
            nn.Linear(512, class_num),
            nn.GELU()
        )

    def forward(self,x1,x2,w1,w2):

        x1,_,_,_,_,shape1 = self.spectral_encoder.encoder_forward(x1,w1,0.)
        x2, _, _, _, _, shape2 = self.spectral_encoder.encoder_forward(x2, w2, 0.)

        B1, W1, H1, C1 = shape1
        B2, W2, H2, C2 = shape2
        x1 = x1.view(B1,H1,W1,-1).permute(0,-1,1,2)
        x2 = x2.view(B2, H2, W2, -1).permute(0, -1, 1, 2)

        x1 = self.convblock(x1)
        x2 = self.convblock(x2)

        x = abs(x1-x2)

        x = self.pool(x).view(B1,-1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    from datautils.readmetadata import readcenterwavelength
    batch_size = 20
    input1 = torch.randn(batch_size, 7, 7, 224)
    input2 = torch.randn(batch_size, 7, 7, 224)

    wave1 = torch.tensor(
        np.expand_dims(np.array(readcenterwavelength('ENMAP01_METADATA.XML')).astype('float'), 0).repeat(batch_size, axis=0))
    wave2 = torch.tensor(
        np.expand_dims(np.array(readcenterwavelength('ENMAP01_METADATA.XML')).astype('float'), 0).repeat(batch_size, axis=0))


    # model_size = ['small','base','large','huge']
    # for size in model_size:
    #     m = Classification(10, size)
    #     torch.save(m, f'{size}.pt')
    m = ChangedetectionModel(2, 'small')

    h = m(input1,input2,wave1,wave2)

    a = 0

