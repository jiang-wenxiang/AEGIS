import math

import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from model.attention_test import  PreNorm, FeedForward,LinFormerAttention
from model.atten111 import ChannelLinFormerAttention,SpLinFormerAttention
import numpy as np
from einops import rearrange
import numbers
import torch.nn.functional as F
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class FFFN(nn.Module):
    def __init__(self, dim, img_size,ffn_expansion_factor):
        super(FFFN, self).__init__()
        self.img_size = img_size
        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=1,bias=False)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1,bias=False)
        self.dim = dim
        self.complex_weight = nn.Parameter(torch.ones(dim*2,img_size, img_size, dtype=torch.float32) )
    def forward(self, x):
        x = rearrange(x, 'b (h w) c ->b c h w ',h=self.img_size,w=self.img_size)
        x = self.project_in(x)
        x = F.gelu(x)*self.complex_weight
        x = self.project_out(x)
        x = rearrange(x, 'b c h w->b (h w) c ')
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, img_size,  heads, dim_head, length,dropout=0., last_stage=False,share_kv=False,ffn_expansion_factor=2.66,):
        super().__init__()
        self.img_size= img_size
        self.norm = nn.LayerNorm(dim)
        self.attn1 = nn.Sequential(PreNorm(dim,ChannelLinFormerAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage,share_kv=share_kv)))
        self.attn2 = nn.Sequential(PreNorm(dim,SpLinFormerAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage,share_kv=share_kv)))
        self.ffn = FFFN(dim,img_size=img_size ,ffn_expansion_factor=ffn_expansion_factor)
        self.ln = nn.Linear(in_features=2*(64+length), out_features=(64+length),bias=False)
    def forward(self, x):
        att1 = self.attn1(x)
        att2 = self.attn2(x)
        attn = torch.cat([att1,att2],dim=-1)
        attn = self.ln(attn)
        x = x + attn
        # x = to_4d(x,self.img_size,self.img_size)
        x = x + self.ffn(self.norm(x))
        # x = to_3d(x)
        return x
class CVTEncoder(nn.Module):
    def __init__(self, image_size, in_channels,  heads , depth,length ,dim=64, pool='cls', dropout=0., emb_dropout=0., scale_dim=4,share_kv=False,):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = in_channels
        #  Convolutional Token Embedding
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, self.dim, 3, 1, 1),
            Rearrange('b c h w -> b (h w) c',h=image_size, w=image_size),
            nn.LayerNorm(self.dim)
        )
        # Convolutions Vision Transformers block
        self.stage1_transformer = nn.Sequential(*[
            TransformerEncoderBlock(dim=self.dim, img_size=image_size, heads=heads, dim_head=self.dim,
                                    dropout=dropout,share_kv=share_kv,length=length)
            for _ in range(depth)
        ],
                                                )
        self.dropout = nn.Dropout(emb_dropout)
        # b c h w 1 3 224 224 -> 1 64 56 56 -> 1 (56 * 56) 64 -> 1 (3136) 64
    def forward(self, img):
        xs = self.stage1_conv_embed(img)
        xs = self.dropout(xs)
        xs = self.stage1_transformer(xs)
        xs = rearrange(xs,'b (h w) c-> b c h w',h=img.shape[2],w=img.shape[3])
        return xs
class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, img_size,  heads, dim_head, dropout=0., last_stage=False,share_kv=False,ffn_expansion_factor=2.66,):
        super().__init__()
        self.img_size= img_size
        self.norm = nn.LayerNorm(dim)
        self.attn1 = nn.Sequential(PreNorm(dim,ChannelLinFormerAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage,share_kv=share_kv)))
        self.attn2 = nn.Sequential(PreNorm(dim,SpLinFormerAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage,share_kv=share_kv)))
        self.ffn = FFFN(dim,img_size=img_size ,ffn_expansion_factor=ffn_expansion_factor)
        self.ln = nn.Linear(in_features=128, out_features=64,bias=False)
    def forward(self, x):
        att1 = self.attn1(x)
        att2 = self.attn2(x)
        attn = torch.cat([att1,att2],dim=-1)
        attn = self.ln(attn)
        x = x + attn
        # x = to_4d(x,self.img_size,self.img_size)
        x = x + self.ffn(self.norm(x))
        # x = to_3d(x)
        return x
class CVTDecoder(nn.Module):
    def __init__(self, image_size, in_channels,  heads , depth,dim=64, pool='cls', dropout=0., emb_dropout=0., scale_dim=4,share_kv=False,):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = in_channels
        #  Convolutional Token Embedding
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, self.dim, 3, 1, 1),
            Rearrange('b c h w -> b (h w) c',h=image_size, w=image_size),
            nn.LayerNorm(self.dim)
        )
        # Convolutions Vision Transformers block
        self.stage1_transformer = nn.Sequential(*[
            TransformerDecoderBlock(dim=self.dim, img_size=image_size, heads=heads, dim_head=self.dim,
                                    dropout=dropout,share_kv=share_kv)
            for _ in range(depth)
        ],
                                                )
        self.dropout = nn.Dropout(emb_dropout)
        # b c h w 1 3 224 224 -> 1 64 56 56 -> 1 (56 * 56) 64 -> 1 (3136) 64
    def forward(self, img):
        xs = self.stage1_conv_embed(img)
        xs = self.dropout(xs)
        xs = self.stage1_transformer(xs)
        xs = rearrange(xs,'b (h w) c-> b c h w',h=img.shape[2],w=img.shape[3])
        return xs

# class CvT(nn.Module):
#     def __init__(self, image_size, in_channels, num_classes, dim=64, kernels=[7, 3, 3], strides=[4, 2, 2],
#                  heads=[1, 3, 6] , depth = [1, 2, 10], pool='cls', dropout=0., emb_dropout=0., scale_dim=4):
#         super().__init__()
#
#
#
#
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         self.pool = pool
#         self.dim = dim
#
#         ##### Stage 1 #######
#         self.stage1_conv_embed = nn.Sequential(
#             nn.Conv2d(in_channels, dim, kernels[0], strides[0], 2),
#             Rearrange('b c h w -> b (h w) c', h = image_size//4, w = image_size//4),
#             nn.LayerNorm(dim)
#         )
#         self.stage1_transformer = nn.Sequential(
#             Transformer(dim=dim, img_size=image_size//4,depth=depth[0], heads=heads[0], dim_head=self.dim,
#                         mlp_dim=dim * scale_dim, dropout=dropout),
#             Rearrange('b (h w) c -> b c h w', h = image_size//4, w = image_size//4)
#         )
#
#         ##### Stage 2 #######
#         in_channels = dim
#         scale = heads[1]//heads[0]
#         dim = scale*dim
#         self.stage2_conv_embed = nn.Sequential(
#             nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
#             Rearrange('b c h w -> b (h w) c', h = image_size//8, w = image_size//8),
#             nn.LayerNorm(dim)
#         )
#         self.stage2_transformer = nn.Sequential(
#             Transformer(dim=dim, img_size=image_size//8, depth=depth[1], heads=heads[1], dim_head=self.dim,
#                         mlp_dim=dim * scale_dim, dropout=dropout),
#             Rearrange('b (h w) c -> b c h w', h = image_size//8, w = image_size//8)
#         )
#
#         ##### Stage 3 #######
#         in_channels = dim
#         scale = heads[2] // heads[1]
#         dim = scale * dim
#         self.stage3_conv_embed = nn.Sequential(
#             nn.Conv2d(in_channels, dim, kernels[2], strides[2], 1),
#             Rearrange('b c h w -> b (h w) c', h = image_size//16, w = image_size//16),
#             nn.LayerNorm(dim)
#         )
#         self.stage3_transformer = nn.Sequential(
#             Transformer(dim=dim, img_size=image_size//16, depth=depth[2], heads=heads[2], dim_head=self.dim,
#                         mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True),
#         )
#
#
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout_large = nn.Dropout(emb_dropout)
#
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, img):
#
#         xs = self.stage1_conv_embed(img)
#         xs = self.stage1_transformer(xs)
#
#         xs = self.stage2_conv_embed(xs)
#         xs = self.stage2_transformer(xs)
#
#         xs = self.stage3_conv_embed(xs)
#         b, n, _ = xs.shape
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         xs = torch.cat((cls_tokens, xs), dim=1)
#         xs = self.stage3_transformer(xs)
#         xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]
#
#         xs = self.mlp_head(xs)
#         return xs


if __name__ == "__main__":
    img = torch.randn([1,64,16,16])
    # model = CVTEncoder(224, 3, 1000)
    model = CVTEncoderDecoder(16,64,
                              depth = 4,
                              heads = 6,
                              length=50)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(img)

    print("Shape of out :", out.shape)  # [B, num_classes]