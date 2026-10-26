import numpy as np
import torch
import torch.nn as nn
import cv2
import math
from torch.autograd import Variable
import torch.nn.functional as F
from kornia.filters import GaussianBlur2d
class Gaussian_blur(nn.Module):
    def __init__(self, size):
        super(Gaussian_blur, self).__init__()
        self.size = size
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        image = image.cpu().detach().numpy() * 255
        image = image.transpose((0, 2, 3, 1))
        for idx in range(image.shape[0]):
            encoded_image = image[idx]
            noise_image = cv2.GaussianBlur(encoded_image, (self.size, self.size), 0)
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        return batch_noise_image / 255


def Gaussian_kernel(sigma,kernel):
    if type(sigma) is str:
        sigma=float(sigma)
    if type(kernel) is str:
        kernel=int(kernel)
    sigma_value=sigma*sigma
    result=torch.zeros((kernel,kernel))
    radius=int(kernel/2)
    for i in range(kernel):
        for j in range(kernel):
            result[i,j]=(1/(math.pi*sigma_value))*(math.exp(-((i-radius)**2+(j-radius)**2)/sigma_value))
    all=result.sum()
    result=result/all
    return result
class Gaussian_blur1(nn.Module):
    def __init__(self, sigma):
        super(Gaussian_blur1, self).__init__()
        self.kernel= 3#int(kernel)
        self.sigma=float(sigma)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noised_and_cover):
        # get the gaussian filter
        encode_image=noised_and_cover[0]
        batch_size,channel=encode_image.shape[0],encode_image.shape[1]
        assert encode_image.shape[1]==3|1
        gaussian_kernel=Gaussian_kernel(self.sigma,self.kernel)
        kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        kernel=kernel.expand(channel,1,self.kernel,self.kernel).to(self.device)
        # print(kernel)
        # print(kernel.size())
        # print(encode_image.size())
        # weight = nn.Parameter(data=kernel, requires_grad=False)

        noised_and_cover[0]=F.conv2d(encode_image,kernel,stride=1,padding=1,groups=3)
        return noised_and_cover[0]

class Gaussian_blur2(nn.Module):

    def __init__(self, sigma, kernel=3):
        super(Gaussian_blur2, self).__init__()
        self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        return self.gaussian_filter(image)