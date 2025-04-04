import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import DropPath
from .model import SpectralSharedEncoder

class ClassificationModel(nn.Module):
    def __init__(self, class_num , model_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_size == 'small':
            self.embedding_dim = 256
            self.spectral_encoder =SpectralSharedEncoder(
                embedding_dim = 256,
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
            nn.Conv2d(64, 512, 3, 2,1), # h,w /2
            nn.BatchNorm2d(512),
            nn.GELU(),
            # nn.Conv2d(128, 256, 2, 2,1), # h,w /4
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 512, 2, 2, 1), # h,w /8
            # nn.BatchNorm2d(512),
            # nn.GELU(),
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,1024),
            nn.Dropout(0.4),
            nn.GELU(),
            nn.Linear(1024, class_num),
            nn.GELU()
        )

    def forward(self,x,wavelength):
        x,_,_,_,_,shape = self.spectral_encoder.encoder_forward(x,wavelength,0.)
        B, W, H, C = shape
        x = x.view(B,H,W,-1).permute(0,-1,1,2)
        x = self.convblock(x)
        x = self.pool(x).view(B,-1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    from datautils.readmetadata import readcenterwavelength
    input1 = torch.randn(1, 5, 5, 224)
    input2 = torch.randn(1, 7, 7, 235)
    inputs = [input1, input2]
    waves = [
        torch.tensor(
        np.expand_dims(np.array(readcenterwavelength('ENMAP01_METADATA.XML')).astype('float'), 0).repeat(1, axis=0)),
        torch.tensor(
        np.expand_dims(np.array(readcenterwavelength('METADATA.XML')).astype('float'), 0).repeat(1, axis=0)),
    ]
    # model_size = ['small','base','large','huge']
    # for size in model_size:
    #     m = Classification(10, size)
    #     torch.save(m, f'{size}.pt')
    m = ClassificationModel(10, 'small')
    for inp, wv in zip(inputs, waves):
        h = m(inp, wv)

    a = 0

