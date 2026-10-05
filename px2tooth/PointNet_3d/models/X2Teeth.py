import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


import torch.nn.parallel
import torch.utils.data
import os
import sys
sys.path.append(os.path.dirname(__file__))
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import chamfer
from torch.autograd import Function
from unet_parts import *
import torch.nn.functional as F
import cv2
import numpy as np

# from unet_parts import *

###################################
#  Unet + PointNet  端到端
# 原始拼接方法
###################################

class get_model(nn.Module):
    def __init__(self, n_classes, n_channels=1,  bilinear=False):
        super(get_model, self).__init__()
        # seg
        self.n_classes = n_classes
        self.n_channels =n_channels
        self.bilinear =bilinear
        self.unet_part = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
        self.pointnet_part = PointNet_3d()


    def forward(self, images, mask_values, mask_original, points):  # (16,9,4096)
        '''seg part'''
        m = images
        logist_3d, feature_seg, = self.unet_part(m)
        '''sub part'''
        tooth_outputs = {}
        points_cat = []
        single_teeth_cat = []
        '''mask_values = 28颗牙齿'''
        for tooth_pixel in mask_values:
            # ################  test
            # tooth_pixel = 235
            # #################
            # single teeth method -创建类
            output = Single_Tooth_Grabbing(mask_original, feature_seg)
            # 调用类 image_px_mean feature map
            single_teeth, single_teeth_mask = output(mask_original, feature_seg, tooth_pixel, images)
            # change
            # single_teeth = single_teeth.repeat(16, 1, 1)
            # points = (1,3,4096)  single_teeth = (1,64,4096)
            # single_point = (1,67,4096)
            single_point = torch.cat((points, single_teeth), dim=1)
            points_cat.append(single_point)
            single_teeth_cat.append(single_teeth_mask)
        points_cat = torch.cat(points_cat, dim=0)
        single_mask_cat = torch.cat(single_teeth_cat, dim=0)  # bs， 67，60,50

        '''3d part'''
        x = single_mask_cat  # 28,67,4096
        tooth_outputs = self.pointnet_part(x, mask_values, tooth_outputs)

        return logist_3d, feature_seg, tooth_outputs


# Unet 不要动
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        # bilinear = 是否采用双线性差值
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature_seg = x
        logits = self.outc(x)
        return logits, feature_seg


def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class PointNet_3d(nn.Module):
    def __init__(self):
        super(PointNet_3d, self).__init__()
        # self.k = num_class
        self.inc_point1 = (DoubleConv(64, 128))
        self.inc_point2 = (DoubleConv(128, 256))
        self.inc_point3 = (DoubleConv(256, 512))
        self.ouc_point1 = (Double3dConv(256, 128))
        self.ouc_point2 = (Double3dConv(128, 64))
        self.ouc_point2 = (Double3dConv(64, 3))
        self.fc = nn.Linear(512 * 20 * 20, 256 * 4 * 4)
        self.upconv1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv1 = nn.Conv3d(128, 128, 3, padding=1)
        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv2 = nn.Conv3d(64, 64, 3, padding=1)
        self.upconv3 = nn.ConvTranspose3d(64, 48, 2, stride=2)
        self.conv3 = nn.Conv3d(48, 48, 1)


        # # self.fc1 = nn.Linear(1024, 512)
        # self.fc = nn.Linear(512 * 80 * 80, output_dim)
        # self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)




    def forward(self, x, mask_values, tooth_outputs):  # (16,3,4096)
        batchsize = x.size()[0]  # 16
        n_pts = x.size()[2]  # 4096
        x = self.inc_point1(x)  # bs 128 20 20
        x = self.inc_point2(x)  # 256
        x = self.inc_point3(x)  # bs，512，20，20
        x = torch.relu(self.fc(x.view(batchsize, -1)))
        # x = x.view(batchsize, 512, 20, 20)  # bs，512，20，20
        # x = torch.relu(self.fc(x))
        x = x.reshape(batchsize, 256, 4, 4)  # 三维
        x = self.upconv1(x.unsqueeze(-1)).squeeze(-1) # bs，128，8，8，2
        x = torch.relu(self.conv1(x))  # [bs, 128, 128, 128, 2]
        x = torch.relu(self.upconv2(x))  # [1, 64, 256, 256, 4]
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.upconv3(x))  # [1, 41, 512, 512, 8]
        x = torch.sigmoid(self.conv3(x))  # [1, 41, 512, 512, 8]
        # change tensor
        new_size = (batchsize, 4096, 3)  # 体素化 512512512
        x = F.interpolate(x, size=new_size, mode='trilinear', align_corners=False)  # mode = 插值方式



        for i, tooth_pixel in enumerate(mask_values):
            # 获取第 i 个 (1, 4996, 3) 数组
            x_i = x[i:i + 1]
            trans_feat_i = x[i:i + 1]
            # 将single_point添加到字典中
            tooth_outputs[tooth_pixel] = {"x": x_i, "trans_feat": trans_feat_i}

        return tooth_outputs


class Single_Tooth_Grabbing(nn.Module):
    def __init__(self, mask_original, feature_seg):
        super(Single_Tooth_Grabbing, self).__init__()
        self.true_masks = mask_original
        self.feature_seg = feature_seg

    def forward(self, mask_original, feature_seg, tooth_pixel, images):
        # print(true_masks)        ###### go on
        bbox, tooth_pixel = get_bbox(mask_original, tooth_pixel)
        # 对全景片进行裁剪bbox
        mask_mean, pre_mask_crop, px_crop, single_teeth_mask = get_px_bbox(feature_seg, bbox, mask_original, images)

        return mask_mean, single_teeth_mask






class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.size_average = True
        self.reduce = True
        self.chamfer_dist = ChamferDist()

    def forward(self, pred, target):
        chamfer_loss = 0.
        constant = 1.
        chamfer = 1.
        chamfer_opposite = 1.
        # change
        # for i in range(15):
        dist1, dist2, idx1, idx2 = self.chamfer_dist(target, pred)
        chamfer_loss += chamfer * (torch.mean(dist1) + chamfer_opposite * torch.mean(dist2))

        return chamfer_loss


class ChamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, _idx1, _idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDist(nn.Module):
    def __init__(self):
        super(ChamferDist, self).__init__()

    def forward(self, input1, input2):
        return ChamferFunction.apply(input1, input2)




""" util Parts of the U-Net model """

class Double3dConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


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
    mask = image_px[:, :, y_min:y_max, x_min:x_max]
    px_crop = images[:, :, y_min:y_max, x_min:x_max]
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
    mask = torch.nn.functional.interpolate(mask, size=(20, 20), mode='bilinear', align_corners=False)

    return mask_mean, pre_mask_crop, px_crop, mask



if __name__ == '__main__':
    model = get_model(13)
    points = torch.rand(16, 4096, 3)
    (model())











