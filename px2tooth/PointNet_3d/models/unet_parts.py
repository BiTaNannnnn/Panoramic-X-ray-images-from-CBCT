""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleUpConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(mid_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 原文中就是转置卷积
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # 防止输入图片不是16的整数倍：拼接的时候大小不一致
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# according mask get box
def get_bbox(mask_original, tooth_pixel):
    # # # 读取二值图像
    # main_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/'
    # img1 = cv2.imread(main_path + 'data/masks/CASE_ID_mask.png', 0)

    # # 假设您的牙齿像素值为[10, 15, 20, 25]
    # tooth_pixels = [21,  55,  60,  65,  75,  80,  85,  90, 105, 110, 115, 125, 130, 135, 155, 160, 165, 175,
    #  180, 185, 190, 205, 210, 215, 225, 230, 235, 240]
    img = mask_original.clone()
    img = torch.squeeze(img)
    img = img.cpu()
    img_np = img.numpy()
    img_np = img_np.astype(np.uint8)

    ################  test
    # tooth_pixel = 235
    #################
    tooth_pixel = int(tooth_pixel)
    # 找到所有等于tooth_pixel的像素点的坐标
    if img_np.ndim == 3:
        img_np = np.mean(img_np, axis=2)
    else:
        pass
    y_coords, x_coords = (img_np == tooth_pixel).nonzero()


    # 计算最小外接矩形
    rect = cv2.minAreaRect(np.column_stack((x_coords, y_coords)))

    # 绘制矩形
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # 垂直的矩形框
    x_min = np.min(box[:, 0])
    y_min = np.min(box[:, 1])
    x_max = np.max(box[:, 0])
    y_max = np.max(box[:, 1])
    box_con = ((x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min))
    box_con = np.array(box_con)
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # 将矩形内部的像素值设置为tooth_pixel
    # 第一个参数是要绘制轮廓的图像，第二个参数是轮廓本身，第三个参数是要绘制的轮廓的索引
    # （在这种情况下，只绘制一个轮廓，所以索引为 0），第四个参数是轮廓的颜色
    # （在这种情况下，为 tooth_pixel），第五个参数是轮廓线条的粗细（在这种情况下，为 2）。
    img_np = img_np.astype(np.uint8)
    # print(img_np.dtype)
    # print(img_np.shape)
    img_box = cv2.drawContours(img_np, [box_con], 0, tooth_pixel, 2)
    # # 显示图像
    # plt.imshow(img_box, cmap='gray')
    # plt.show()
    # print('finished -- ID ：CASE_ID', )
    return box_con, tooth_pixel
    # # save_path = main_path + 'Data_Tooth/label_box/CASE_ID_box.png'
    # # cv2.imwrite(save_path, img_box)

    # return box_con



def get_px_bbox(feature_seg, box_con, image_mask, images):
    # # 读取两张图片
    # # image_px = 全景片原图
    # image_px1 = cv2.imread('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train/imgs/CASE_ID.png')
    # print(image_px1.dtype)
    # print(image_px1.shape)

    image_px = feature_seg
    # shape = （1, 64, 512, 512）

    # 裁剪  bbox
    x_min = np.min(box_con[:, 0])
    y_min = np.min(box_con[:, 1])
    x_max = np.max(box_con[:, 0])
    y_max = np.max(box_con[:, 1])
    '''trans unet 实验添加'''
    mask = image_px[:, :, y_min:y_max, x_min:x_max]
    px_crop = images[:, :, y_min:y_max, x_min:x_max]
    if x_max - x_min == 0:
        mask = image_px[:, :, y_min-1:y_max+1, x_min-1:x_max+1]
        px_crop = images[:, :, y_min-1:y_max+1, x_min-1:x_max+1]
    if y_max - y_min == 0:
        mask = image_px[:, :, y_min-1:y_max+1, x_min-1:x_max+1]
        px_crop = images[:, :, y_min-1:y_max+1, x_min-1:x_max+1]


    # # show
    # image_show = px_crop.squeeze()
    # image_show = image_show.cpu()
    # image_show = image_show.detach().numpy()
    # image_show = image_show.transpose(1, 2, 0)
    # plt.imshow(image_show)
    # plt.show()

    pre_mask_crop = image_mask
    mask_mean = torch.mean(mask, dim=(2, 3), keepdim=True)
    # shape= (1, 64, 1, 1)
    # image_px_mean = image_px_mean.repeat(1, 1, 4096, 1).squeeze()
    mask_mean = mask_mean.repeat(1, 1, 4096, 1)  # 1，64，4096，1
    mask_mean = mask_mean.squeeze(3)  # 1，64，4096

    '''mask precess'''
    #
    # _, _, w, h = mask.size()
    #
    # if h == 0:
    #     print("W 不满足条件，使用原本 mask 的 H 值创建空白 tensor")
    #     mask_zero = torch.zeros((1, 67, w, 80), dtype=mask.dtype, device=mask.device)
    #     mask_zero[:, :, :h, :] = mask[:, :, :w, :]  # 将原本 mask 的高度赋值到创建的空白 tensor 中
    #     mask = torch.nn.functional.interpolate(mask, size=(80, 80), mode='bilinear', align_corners=False)
    # else:
    #     # 进行插值
    #     mask = torch.nn.functional.interpolate(mask, size=(80, 80), mode='bilinear', align_corners=False)
    mask = torch.nn.functional.interpolate(mask, size=(80, 80), mode='bilinear', align_corners=False)

    return mask_mean, pre_mask_crop, px_crop, mask
