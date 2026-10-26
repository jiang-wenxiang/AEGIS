import torch
import torch.nn as nn
import network.module_util as mutil
import torch.nn.functional as F

class SAFM(nn.Module):
    def __init__(self, dim,  bias=True): # NoDWT:1      YesDWT:4
        super().__init__()
        n_levels=4
        #     n_levels=1
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        self.conv1 = nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(chunk_dim * 2, chunk_dim, 3, 1, 1, bias=bias)
        # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)

        # Activation
        self.lrelu = nn.LeakyReLU(inplace=True)# self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s1 = self.conv1(s)
                s2 = self.conv2(torch.cat((s, s1), dim=1))
                s_out = F.interpolate(s2, size=(h, w), mode='nearest')
            else:
                s1 = self.conv1(xc[i])
                s_out = self.conv2(torch.cat((xc[i], s1), dim=1))
            out.append(s_out)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.lrelu(out) * x
        return out
class CCM(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        mutil.initialize_weights([self.conv2], 0.)
    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return out
class FMM(nn.Module):
    def __init__(self, input, bias=True):
        super().__init__()
        self.safm = SAFM(input, bias=bias)
        self.ccm = CCM(input, bias=bias)
    def forward(self, x):
        out = self.safm(x) + x
        out = self.ccm(out) + out
        return out