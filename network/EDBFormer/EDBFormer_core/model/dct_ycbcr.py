from tkinter import Y
import numpy
import numpy as np
from numpy.core.fromnumeric import shape
# from sympy import im
# from sklearn.feature_extraction import image
import torch

from PIL import Image, ImageOps
import time
from multiprocessing.dummy import Pool as ThreadPool
from einops import rearrange, repeat


def ycbcr2rgb(input_im, cuda=False):
   
    im_flat = input_im.contiguous().view(-1, 3).float()
   
    y = torch.tensor([219, 224, 224])
    bias = torch.tensor([16.0, 128.0, 128.0])
    mat = torch.tensor([
        [1.00000000e00, -1.21889419e-06, 1.40199959e00],
        [1.00000000e00, -3.44135678e-01, -7.14136156e-01],
        [1.00000000e00, 1.77200007e00, 4.06298063e-07],
    ])
    if cuda:
        y = y.cuda()
        bias = bias.cuda()
        mat = mat.cuda()
    im_flat = im_flat - bias
    im_flat = im_flat / y
    temp = im_flat.mm(torch.transpose(mat, 0, 1))
    out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])
    return out


def rgb2ycbcr(input_im, cuda=False):
    im_flat = input_im.contiguous().view(-1, 3).float()   # (H*W,3)
    mat = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ])
    y = torch.tensor([219, 224, 224])
    bias = torch.tensor([16.0, 128.0, 128.0])
    if cuda:
        mat = mat.cuda()
        y = y.cuda()
        bias = bias.cuda()
    temp = im_flat.mm(torch.transpose(mat, 0, 1))
    temp = temp * y
    temp = temp + bias
    out = temp.view(3, input_im.shape[1], input_im.shape[2])
    return out

def my_rgb2ycbcr(im):
    # im (3,H,W)
    r = im[0,:,:].cuda()  #(H,W)
    g = im[1,:,:].cuda()
    b = im[2,:,:].cuda()

    Y = (r*65.481 + g*128.553 + b*24.966 + 16.0).unsqueeze(0)
    Cb = (r*-37.797 - g*74.203 + b*112.0 + 128.0).unsqueeze(0)
    Cr = (r*112.0 - g*93.786 - b*18.214 + 128.0).unsqueeze(0)

    YCbCr = torch.cat([Y, Cb, Cr], dim=0)

    return YCbCr

def my_ycbcr2rgb(ycbcr):
    Y = ycbcr[0,:,:].cuda()  #(H,W)
    Cb = ycbcr[1,:,:].cuda()
    Cr = ycbcr[2,:,:].cuda()

    Y = (Y - 16.0) / 219.0
    Cb = (Cb - 128.0) / 224.0
    Cr = (Cr - 128.0) / 224.0
 
    R = (Y*1.00000000e00 + Cb*-1.21889419e-06 + Cr*1.40199959e00).unsqueeze(0)
    G = (Y*1.00000000e00 + Cb*-3.44135678e-01 + Cr*-7.14136156e-01).unsqueeze(0)
    B = (Y*1.00000000e00 + Cb*1.77200007e00 + Cr*4.06298063e-07).unsqueeze(0)

    RGB = torch.cat([R,G,B],dim=0)
    return RGB


def rgb2ycbcr_batch(images, cuda=False):
    # im (3,H,W)
    r = images[:,0,:,:].cuda()  #(N,H,W)
    g = images[:,1,:,:].cuda()
    b = images[:,2,:,:].cuda()

    Y = (r*65.481 + g*128.553 + b*24.966 + 16.0).unsqueeze(1)
    Cb = (r*-37.797 - g*74.203 + b*112.0 + 128.0).unsqueeze(1)
    Cr = (r*112.0 - g*93.786 - b*18.214 + 128.0).unsqueeze(1)

    YCbCr = torch.cat([Y, Cb, Cr], dim=1)

    return YCbCr

def ycbcr2rgb_batch(images, cuda=False):
    Y = images[:,0,:,:].cuda()  #(N,H,W)
    Cb = images[:,1,:,:].cuda()
    Cr = images[:,2,:,:].cuda()

    Y = (Y - 16.0) / 219.0
    Cb = (Cb - 128.0) / 224.0
    Cr = (Cr - 128.0) / 224.0
 
    R = (Y*1.00000000e00 + Cb*-1.21889419e-06 + Cr*1.40199959e00).unsqueeze(1)
    G = (Y*1.00000000e00 + Cb*-3.44135678e-01 + Cr*-7.14136156e-01).unsqueeze(1)
    B = (Y*1.00000000e00 + Cb*1.77200007e00 + Cr*4.06298063e-07).unsqueeze(1)

    RGB = torch.cat([R,G,B],dim=1)
    return RGB


  



# def dct_batch(imgs):
#     return (dct.dct_2d(imgs.squeeze(1))).unsqueeze(1)


# def idct_batch(imgs):
#     return (dct.idct_2d(imgs.squeeze(1))).unsqueeze(1)


import math

def getDCTM(size):
    h = size[0]
    w = size[1]
    dctm = torch.ones((h,w))
    for i in range(h):
        for j in range(w):
            c = math.sqrt(1.0/h) if i==0 else math.sqrt(2.0/h)
            dctm[i, j] = c * math.cos( (j + 0.5) * math.pi * i / h)
  
    return dctm.cuda()

dctm = getDCTM((8,8))
dctm_T = dctm.transpose(0, 1)
dctm_inv = torch.inverse(dctm)   # dctm.inverse()
dctm_T_inv = dctm_T.inverse() 
   
def my_dct2d(input, dctm=dctm, dctm_T=dctm_T):
    # input (b,8,8)
    dctm = dctm.unsqueeze(0)
    dctm = dctm.expand(input.shape)
    dctm_T = dctm_T.unsqueeze(0)
    dctm_T = dctm_T.expand(input.shape)

    temp = torch.bmm(dctm, input)
    ans = torch.bmm(temp, dctm_T)
    return ans
    
def my_idct2d(input, dctm_inv=dctm_inv, dctm_T_inv=dctm_T_inv):
    # input (b,8,8)
    dctm_inv = dctm_inv.unsqueeze(0)
    dctm_inv = dctm_inv.expand(input.shape)
    dctm_T_inv = dctm_T_inv.unsqueeze(0)
    dctm_T_inv = dctm_T_inv.expand(input.shape)

    temp = torch.bmm(dctm_inv, input)
    ans = torch.bmm(temp, dctm_T_inv)
    return ans





def dct_dxc(imgs):
    # imgs = imgs.squeeze(1)
    item_list = rearrange(imgs, 'b (h p1) (w p2) -> (b h w) p1 p2', p1=8, p2=8)
    item_list = my_dct2d(item_list)
    item_list = rearrange(item_list, '(b h w) p1 p2 -> b (p1 p2) h w', h=imgs.shape[1]//8, w=imgs.shape[2]//8)
    # item_list = item_list.unsqueeze(1)
    return item_list
    
def idct_dxc(imgs):
    # imgs = imgs.squeeze(1)
    item_list = rearrange(imgs, 'b (p1 p2) h w -> (b h w) p1 p2', p1=8, p2=8)
    item_list = my_idct2d(item_list)
    item_list = rearrange(item_list, '(b h w) p1 p2 -> b (h p1) (w p2)', h=imgs.shape[2], w=imgs.shape[3])
    # item_list = item_list.unsqueeze(1)
    return item_list


def dwt(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2


    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h




if __name__ == '__main__':
    # img_cover = Image.open('/media/whh/My Passport/workplaceOfZZW/StegaStamp_pytorch/data/img/test/im2.jpg')
    img_cover = Image.open('/media/whh/My Passport/workplaceOfZZW/HiDDeN/save4.jpg')
    img_cover = ImageOps.fit(img_cover, (400, 400))
    img_cover = np.array(img_cover, dtype=np.float32)/255.
    img_cover = np.transpose(img_cover, (2, 0, 1))

    img_cover = torch.from_numpy(img_cover)
    print("img", img_cover)

    ycbcr = my_rgb2ycbcr(img_cover)
    print(ycbcr)
    # ran_y = (torch.clamp(torch.randn(400,400),0.01,0.99)*255.).unsqueeze(0).cuda()
    # cb = ycbcr[1,:,:].unsqueeze(0)
    # cr = ycbcr[2,:,:].unsqueeze(0)
    # new = torch.cat([ran_y, cb, cr], dim=0)
    # print(new)

    # img_cover = my_ycbcr2rgb(new)

    # img = img_cover.cpu().numpy() * 255.
    # img = img.transpose(1,2,0)
    # print("img_size", img.shape)
    # im = Image.fromarray(img.astype(np.uint8))
    # im.save("save4.jpg")


    
    # img = dct_3d_88(img_cover)
    # print("img_dct", img)
    #
    # img_rgb = idct_3d_88(img)
    # print("img_rgb", img_rgb)

    # img_rgb = dct.idct_3d(img)
    # print("img_rgb", img_rgb)


    # img = numpy.expand_dims(img_cover, axis=0)
    # img1 = numpy.concatenate([img, img], axis=0)
    # img1 = torch.from_numpy(img1)
    # print("img", img1)

    # img_batch = rgb2ycbcr_batch(img1) / 255.
    # print("img_ycbcr", img_batch)
    # print("img_ycbcr_max", torch.max(img_batch))
    # print("img_ycbcr_min", torch.min(img_batch))

    # img_batch = dct_batch(img_batch)
    # print("img_ycbcr_dct", img_batch)

    # img_batch = idct_batch(img_batch)
    # print("img_ycbcr_dct_idct", img_batch)

    # img_batch = img_batch * 255.
    # img = ycbcr2rgb_batch(img_batch)
    # print("img_ycbcr_dct_idct_iycbcr", img)

    # img = img.numpy()
    # img = img[0]
    # img = img.transpose(1,2,0)
    # print("img_size", img.shape)

    # im = Image.fromarray(img.astype(np.uint8))
    # im.save("save4.jpg")

    pass

