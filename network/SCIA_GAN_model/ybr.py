import torch.nn.functional as F
import torch
import torch.nn as nn
import config as c
import numpy as np
from network.dwt_dct import DWTForward_Init#, DWTInverse_Init, DWT, IWT  #新加
import skimage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
loss_fn = torch.nn.MSELoss(reduction='mean')

# separate the ycbcr images for the loss calculation
def ycbcr_images(images,encoded_images):
    cover_ybr = skimage.color.rgb2ycbcr(images.permute(0, 2, 3, 1).cpu().detach().numpy())
    cover_ybr = torch.from_numpy(cover_ybr / 255.0).permute(0, 3, 1, 2)
    stego_ybr = skimage.color.rgb2ycbcr(encoded_images.permute(0, 2, 3, 1).cpu().detach().numpy())
    stego_ybr = torch.from_numpy(stego_ybr / 255.0).permute(0, 3, 1, 2)
    dwtimage = DWTForward_Init(J=1, mode='zero', wave='haar').to(device)
    # idwtimage = DWTInverse_Init(mode='zero', wave='haar').to(device)
    stego_Yl, stego_Yh = dwtimage(stego_ybr[:, 0, :, :].unsqueeze(1).to(device))
    cover_Yl, cover_Yh = dwtimage(cover_ybr[:, 0, :, :].unsqueeze(1).to(device))
    stego_color = torch.cat((stego_ybr[:, 1, :, :], stego_ybr[:, 2, :, :]), 1).unsqueeze(1)
    cover_color = torch.cat((cover_ybr[:, 1, :, :], cover_ybr[:, 2, :, :]), 1).unsqueeze(1)
    # stego_YH = torch.tensor([item.cpu().detach().numpy() for item in stego_Yh]).to(device).squeeze()
    # cover_YH = torch.tensor([item.cpu().detach().numpy() for item in cover_Yh]).to(device).squeeze()
    stego_YH = torch.tensor(np.array([item.cpu().detach().numpy() for item in stego_Yh])).to(device).squeeze()
    cover_YH = torch.tensor(np.array([item.cpu().detach().numpy() for item in cover_Yh])).to(device).squeeze()
    return cover_ybr, stego_ybr, stego_Yl, stego_Yh, cover_Yl, cover_Yh, stego_color, cover_color, stego_YH, cover_YH

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
class GWLoss(nn.Module):
    """Gradient Weighted Loss"""
    def __init__(self, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = c.w
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)
def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)
def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)

