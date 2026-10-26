import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from .DiffJPEG.DiffJPEG import DiffJPEG


from torch.utils.data import DataLoader
from utils.Dataloader import MBRSDataset
import torchvision

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
patch_size = '128'  # The crop size from the original image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # print(type(x.to(device)))
        x = self.conv(x.to(device))
        # print(type(x))
        return x.to(device)


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# The network used for OSN noise modeling
class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, isResidual=True, isJPEG=True):
        super(U_Net, self).__init__()
        self.name = 'U_Net'
        self.isResidual = isResidual
        self.isJPEG = isJPEG

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Tanh()

        if self.isJPEG:
            self.diff_jpeg = DiffJPEG(height=int(patch_size), width=int(patch_size), differentiable=True)

    def forward(self, x, quality=50):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        if self.isResidual:
            out = 0.02 * self.active(out) + 0.98 * x
        else:
            out = self.active(out)
        if self.isJPEG:
            out = self.diff_jpeg((out + 1) / 2, quality=quality)
            out = (out - 0.5) * 2
        return out

# model = U_Net()
# pretrained = torch.load('weights/OSN_UNet_weights.pth')
# model.load_state_dict(pretrained)
# model.eval()
# test_transform = transforms.Compose([
#     np.float32,
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
#
# img = cv2.imread('data/osn/original.png')
# orig = img
# img = img.astype('float') / 255.
# img = test_transform(img).unsqueeze(0)
# print(img.shape)
# # OSN Noise Modeling
# img = model(img)
#
# img = img[0].permute(1, 2, 0)
# img = img.detach().cpu().numpy()
# img = img * 127.5 + 127.5
# cv2.imwrite('data/osn/simulated_osn.png', np.uint8(img))
#
# resdual = img - orig
# if True:
#     # For better visualization
#     resdual[resdual > 8] = 0
#     resdual[resdual < -8] = 0
# resdual = cv2.normalize(resdual, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
# resdual = cv2.cvtColor(resdual[:, :, :3], cv2.COLOR_RGB2GRAY)
# cv2.imwrite('data/osn/residual.png', resdual)
# print("ok")

def osn(original_image):
    # print('osn')
    model = U_Net().to(device)
    pretrained = torch.load('/media/WRL/D/WRL/MBRS-main/network/noise_layers/weights/OSN_UNet_weights.pth')
    model.load_state_dict(pretrained)
    model.isJPEG=False
    model.eval()
    original = original_image
    osn = model(original)
    residual = osn - original
    return original, osn, residual

class Osn(nn.Module):
    def __init__(self):
        super(Osn, self).__init__()
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        original_image, osn_image, residual_image = osn(image)
        return osn_image
# if __name__ == '__main__':
#     test_dataset = MBRSDataset(os.path.join("/home/WRL/WRL/ZYMSJCode/ZYMSJcode_改/Datasets/", "validation"), 256, 256)
#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
#     for i, images in enumerate(test_dataloader):
#         original_image, osn_image, residual_image = osn(images)
#         torchvision.utils.save_image(original_image, 'original_image.png')
#         torchvision.utils.save_image(osn_image, 'osn_image.png')
#         torchvision.utils.save_image(residual_image, 'residual_image.png')