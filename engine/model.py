import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import DropPath


# from pos import WavePositionalEncoding


def aggregate(x):
    return x[:, 0, :].unsqueeze(dim=1) + nn.AdaptiveAvgPool1d(1)(x[:, 1:, :].transpose(2, 1)).transpose(2, 1)


class SpectralEmbedding(nn.Module):
    def __init__(self,
                 in_channels,
                 embedding_dim,
                 kernel_sizes: list,
                 ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, embedding_dim, kernel_size, stride=max(kernel_sizes)),
                nn.BatchNorm1d(embedding_dim),
                nn.GELU()) for kernel_size in kernel_sizes])
        self.embedding_out_block = nn.Sequential(
            nn.Conv1d(embedding_dim * 3, embedding_dim, 3, stride=1, padding='same'),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU())

    def forward(self, x):
        L = x.shape[-1]
        padding = math.ceil(x.shape[-1] / max(self.kernel_sizes)) * max(self.kernel_sizes) - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, padding), "constant", 0)
        for block, size in zip(self.embedding_blocks, self.kernel_sizes):
            slice_ = int((max(self.kernel_sizes) - size) / 2)
            _ = block(x[:, :, slice_:])
        x = torch.cat([block(x) for block in self.embedding_blocks], dim=1)
        x = torch.cat([block(x[:, :, int((max(self.kernel_sizes) - size) / 2):])
                       for block, size in zip(self.embedding_blocks, self.kernel_sizes)], dim=1)
        x = self.embedding_out_block(x)
        return x, L


class SpectralSharedEncoder(nn.Module):
    def __init__(self,
                 embedding_dim=1024,
                 max_band=500,
                 encoder_depth=32,
                 decoder_depth=16,
                 num_heads=8,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_band = max_band
        self.global_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.embedding_layer_1 = nn.Sequential(
            nn.Conv1d(1, embedding_dim, 5, stride=1, padding='same'),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(), )

        self.embedding_layer_2 = nn.Sequential(
            nn.Conv1d(1, embedding_dim, 9, stride=1, padding='same'),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU())

        self.embedding_layer_3 = nn.Sequential(
            nn.Conv1d(1, embedding_dim, 13, stride=1, padding='same'),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU())

        self.embedding_layer_4 = nn.Sequential(
            nn.Conv1d(embedding_dim * 3, embedding_dim, 1, stride=1, padding='same'),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU())
        # pos tokenizer
        # self.pos_embed_layer = WavePositionalEncoding(d_model=embedding_dim, max_len=max_band)
        self.div_term = nn.Parameter(torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(100.0) / embedding_dim)),
                                     requires_grad=False)
        # self.max_band = max_band
        self.pe = nn.Parameter(torch.zeros(1, max_band, embedding_dim), requires_grad=False)
        self.encoder_blocks = nn.ModuleList([
            Block(embedding_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth - 1)])

        self.transformer_encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, mlp_ratio, qkv_bias=True,
                                                                 qk_scale=None, norm_layer=norm_layer)
        self.corss_attn = Corss_Attention(embedding_dim, num_heads=num_heads, q_bias=True, q_scale=None)

        self.decoder_blocks = nn.ModuleList([
            Block(embedding_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth - 1)])

        self.norm1 = norm_layer(embedding_dim)
        self.norm2 = norm_layer(embedding_dim)
        self.decoder_layer = nn.Sequential(
            nn.Conv1d(embedding_dim, 1, 3, stride=1, padding='same'),
            # nn.BatchNorm1d(embedding_dim),
            nn.Sigmoid())

        self.initialize_weights()
        print("model initialized")

    def initialize_weights(self):
        torch.nn.init.normal_(self.global_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def embedding(self, x):
        x1 = self.embedding_layer_1(x)
        x2 = self.embedding_layer_2(x)
        x3 = self.embedding_layer_3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.embedding_layer_4(x)
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def encoder_forward(self, x, wave, mask_ratio):
        shape = x.shape  # (B,1,C) or (B,H,W,C)
        if x.dim() == 3:
            x = self.embedding(x)
            wave = ((wave - 400) / 10).unsqueeze(-1)
        elif x.dim() == 4:
            x = self.embedding(x.reshape(-1, 1,shape[-1]))
            # (B,Wave) -> (B,1,Wave) -> (B,H*W,Wave) -> (B,H*W,Wave,1) -> (B*H*W,Wave,1)
            wave = ((wave - 400) / 10).unsqueeze(1).repeat(1, shape[1] * shape[2], 1).unsqueeze(-1).view(-1, shape[-1], 1)
        x = x.transpose(2, 1)
        # 计算中心波段的pos embedding tokens

        pos_tokens = self.pe.repeat(x.shape[0], 1, 1)
        pos_tokens[:, :x.shape[1], 0::2] = torch.sin(self.div_term * wave)
        pos_tokens[:, :x.shape[1], 1::2] = torch.cos(self.div_term * wave)
        x = x + pos_tokens[:, :x.shape[1]]
        # TODO： 增加随机掩码
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # 添加全局变量
        global_tokens = self.global_token.expand(x.shape[0], -1, -1)
        x = torch.cat((global_tokens, x), dim=1)
        # Encoder层
        for block in self.encoder_blocks:
            x = block(x)
        # 输出最后一层encoder层的key和value以及token
        x, k, v = self.transformer_encoder_layer(x)
        z = aggregate(x)
        # z.view(B,H,W,-1) spatial feature
        return z, k, v, ids_restore, pos_tokens, shape

    def decoder_forward(self, z, k, v, ids_restore, pos_tokens):

        # 进行交叉注意力解聚合，作为解码器的输入
        x = self.corss_attn(z, k, v)  # No g
        x = self.norm1(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = x + pos_tokens[:, :x.shape[1]]
        # 进入解码器模块，解码重构原始光谱
        for block in self.decoder_blocks:
            x = block(x)
        x = self.norm2(x).transpose(2, 1)
        x = self.decoder_layer(x)
        return x

    def forward(self, x, wave, mask_ratio):
        z, k, v, ids_restore, pos_tokens,_ = self.encoder_forward(x, wave, mask_ratio)
        x = self.decoder_forward(z, k, v, ids_restore, pos_tokens)
        return z, x

    # def forward(self, x, wave, mask_ratio):
    #     # 输入层多尺度embedding
    #     x = self.embedding(x)
    #     x = x.transpose(2, 1)
    #     # 计算中心波段的pos embedding tokens
    #     # pos_tokens = self.pos_embed_layer(wave)
    #     wave = ((wave - 400) / 10).unsqueeze(-1)
    #     pos_tokens = self.pe.repeat(wave.shape[0], 1, 1)
    #     pos_tokens[:, :wave.shape[1], 0::2] = torch.sin(self.div_term * wave)
    #     pos_tokens[:, :wave.shape[1], 1::2] = torch.cos(self.div_term * wave)
    #     x = x + pos_tokens[:, :x.shape[1]]
    #     # TODO： 增加随机掩码
    #     x, mask, ids_restore = self.random_masking(x, mask_ratio)
    #     # 添加全局变量
    #     global_tokens = self.global_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((global_tokens, x), dim=1)
    #     # Encoder层
    #     for block in self.encoder_blocks:
    #         x = block(x)
    #     # 输出最后一层encoder层的key和value以及token
    #     x, k, v = self.transformer_encoder_layer(x)
    #     # 对输出token聚合
    #     z = aggregate(x)
    #     # 进行交叉注意力解聚合，作为解码器的输入
    #     x = self.corss_attn(z, k, v)  # No g
    #     x = self.norm1(x)
    #     # append mask tokens to sequence
    #     mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    #     x_ = torch.cat([x, mask_tokens], dim=1)
    #     x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #
    #     x = x + pos_tokens[:, :x.shape[1]]
    #     # 进入解码器模块，解码重构原始光谱
    #     for block in self.decoder_blocks:
    #         x = block(x)
    #     x = self.norm2(x).transpose(2, 1)
    #     x = self.decoder_layer(x)
    #     return z, x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x, k, v = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, k, v


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k, v


class Corss_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, q_bias=False, q_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = q_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim, bias=q_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k, v):
        B, N, C = x.shape
        q = self.qkv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        k, v = [k[:, :, 1:, :], v[:, :, 1:, :]]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn.transpose(-1, -2) * v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    from datautils.readmetadata import readcenterwavelength

    # e = SpectralEmbedding(1, 512, [3, 5, 9])
    model = SpectralSharedEncoder(128, encoder_depth=8, decoder_depth=4, num_heads=8)

    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))

    from thop import profile

    input1 = torch.randn(100, 1, 235)
    input2 = torch.randn(100, 235)
    input3 = 0.
    flops, params = profile(model, inputs=(input1,input2,input3))

    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

    # conv1d = nn.Conv1d(1, 20, 3, 9)
    # input1 = torch.randn(20, 1, 224)

    # input1 = torch.randn(20, 2, 2, 224)
    # input2 = torch.randn(20, 2, 2, 235)
    # # padding = math.ceil(input1.shape[-1] / 9) * 9 - input1.shape[-1]
    # # x = torch.nn.functional.pad(input1, (0, padding), "constant", 0)
    # # _ = e(input1)
    # # output = conv1d(x[:, :, 3:])
    # inputs = [input1, input2]
    # waves = [torch.tensor(
    #     np.expand_dims(np.array(readcenterwavelength('ENMAP01_METADATA.XML')).astype('float'), 0).repeat(20, axis=0)),
    #          torch.tensor(
    #              np.expand_dims(np.array(readcenterwavelength('METADATA.XML')).astype('float'), 0).repeat(20, axis=0)),
    #          #
    #          ]
    # # waves = [torch.tensor(np.array(readcenterwavelength('ENMAP01_METADATA.XML')).astype('float')),
    # #          torch.tensor(np.array(readcenterwavelength('METADATA.XML')).astype('float')),
    #
    # # ]
    # for inp, wv in zip(inputs, waves):
    #     h, x_hat = m(inp, wv, .5)
    # pool = nn.AdaptiveAvgPool2d(1)
    # outpu = pool(input1)
    # a = 0
