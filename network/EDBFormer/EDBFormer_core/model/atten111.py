import torch
from torch import nn,einsum
from einops import rearrange

import math
def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor
class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class  ChannelLinFormerAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0., last_stage=False,
                 one_kv_head=False,share_kv=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        n = int(self.img_size*self.img_size)
        self.k= int(n/4)
        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.proj_k =nn.Parameter(init_(torch.zeros(n,self.k)))
        self.share_kv = share_kv
        if not share_kv:
            self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)
            self.proj_v = nn.Parameter(init_(torch.zeros(n,self.k)))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d, h,k = *x.shape, self.heads,self.k
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)
        proj_seq_len = lambda args: torch.einsum('bhnd,nk->bhkd', *args)
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        key = self.to_k(x)
        key = rearrange(key, 'b (h d) l w -> b h (l w) d', h=h)
        # [1,16,256,164] b h n d -> b h k d
        v = self.to_v(x) if not self.share_kv else key
        if not self.share_kv:
            v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        # [1,16,256,164] b h n d -> b h k d [1,16,k,164]
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
        key, v = map(proj_seq_len, zip((key, v), kv_projs))
        # if self.last_stage:
        #     q = torch.cat((cls_token, q), dim=2)
        #     v = torch.cat((cls_token, v), dim=2)
        #     k = torch.cat((cls_token, k), dim=2)
        # [b,h,n,d] [b,h,k,d]->b h n k
        dots = einsum('bhnd,bhkd->bhnk', q, key) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('bhnk,bhkd->bhnd', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
class  SpLinFormerAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0., last_stage=False,
                 one_kv_head=False,share_kv=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        n = int(self.img_size*self.img_size)
        self.k= int(dim/4)
        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.proj_k =nn.Parameter(init_(torch.zeros(dim,self.k)))
        self.share_kv = share_kv
        if not share_kv:
            self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)
            self.proj_v = nn.Parameter(init_(torch.zeros(dim,self.k)))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d, h,k = *x.shape, self.heads,self.k
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)
        proj_seq_len = lambda args: torch.einsum('bhdn,dk->bhkn', *args) #
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h d (l w)', h=h)  # b h d n

        key = self.to_k(x)
        key = rearrange(key, 'b (h d) l w -> b h d (l w) ', h=h) # b h d n
        # [1,16,256,164] b h n d -> b h k d
        v = self.to_v(x) if not self.share_kv else key
        if not self.share_kv:
            v = rearrange(v, 'b (h d) l w -> b h d (l w)', h=h)
        # [1,16,256,164] b h n d -> b h k d [1,16,k,164]
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
        key, v = map(proj_seq_len, zip((key, v), kv_projs))
        # if self.last_stage:
        #     q = torch.cat((cls_token, q), dim=2)
        #     v = torch.cat((cls_token, v), dim=2)
        #     k = torch.cat((cls_token, k), dim=2)
        # [b,h,n,d] [b,h,k,d]->b h n k
        dots = einsum('bhdn,bhkn->bhdk', q, key) * self.scale #k b h n d b h k d

        attn = dots.softmax(dim=-1)

        out = einsum('bhdk,bhkn->bhdn', attn, v)
        out = rearrange(out, 'b h d n -> b n (h d)')
        out =  self.to_out(out)
        return out
