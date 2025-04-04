import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import DropPath
from .model import SpectralSharedEncoder


class SumToOne(nn.Module):
    def __init__(self, scale=1):
        super(SumToOne, self).__init__()
        self.scale = scale
    def forward(self, x):
        x = F.softmax(self.scale * x, dim=1)
        return x

class weightConstraint(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if hasattr(module, 'weight'):
           # print("Entered")
            w = module.weight.data
            w = torch.clamp_min(w, 0)
            module.weight.data = w

class Decoder(nn.Module):
    def __init__(self, num_em, out_channels, kernel_size):
        super(Decoder, self).__init__()
        padding = kernel_size //2
        self.decoder =nn.Conv2d(in_channels=num_em,
                                out_channels=out_channels,
                                kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.relu = nn.ReLU()

    def forward(self, code):
        code = self.relu(self.decoder(code))
        return code

    def getEndmembers(self):
        constraints = weightConstraint()
        self.decoder.apply(constraints)
        return self.decoder.weight.data

class UnmixingModel(nn.Module):
    def __init__(self, num_em, channels, model_size):
        super().__init__()
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
            nn.Conv2d(64, 64, 3, 1,1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # h,w /2
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_em, 3, 1, 1),  # h,w /2
            nn.BatchNorm2d(num_em),
            nn.GELU(),
        )
        self.sumtoone = SumToOne(1)
        self.decoder = Decoder(num_em, channels, kernel_size=1)

    def getAbundances(self,x, wavelength):
        x, _, _, _, _, shape = self.spectral_encoder.encoder_forward(x, wavelength, 0.)
        B, W, H, C = shape
        x = x.view(B, H, W, -1).permute(0, -1, 1, 2)
        abunds = self.convblock(x)
        abunds = self.sumtoone(abunds)
        return abunds

    def forward(self,x, wavelength):
        abunds = self.getAbundances(x, wavelength)
        output = self.decoder(abunds)
        return abunds, output

    def getEndmembers(self):
        endmembers = self.decoder.getEndmembers()
        if endmembers.shape[2] > 1:
            endmembers = np.squeeze(endmembers).mean(axis=2).mean(axis=2)
        else:
            endmembers = np.squeeze(endmembers)
        return endmembers

if __name__ == '__main__':
    from datautils.readmetadata import readcenterwavelength
    input1 = torch.randn(10, 5, 5, 224)
    input2 = torch.randn(10, 7, 7, 235)
    inputs = [input1, input2]
    waves = torch.tensor(
        np.expand_dims(np.array(readcenterwavelength('ENMAP01_METADATA.XML')).astype('float'), 0).repeat(10, axis=0))

    # model_size = ['small','base','large','huge']
    # for size in model_size:
    #     m = Classification(10, size)
    #     torch.save(m, f'{size}.pt')
    m = UnmixingModel(4, 224,'small')
    h = m(input1, waves)


    a = 0

