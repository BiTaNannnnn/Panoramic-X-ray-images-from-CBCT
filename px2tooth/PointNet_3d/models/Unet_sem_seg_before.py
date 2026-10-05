import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import chamfer
from torch.autograd import Function

from unet_parts import *

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

        # 3d
        # self.k = num_class
        self.k = 3
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=67)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, images, mask_values, mask_original, points):  # (16,9,4096)
        # seg
        m = images
        m1 = self.inc(m)
        m2 = self.down1(m1)
        m3 = self.down2(m2)
        m4 = self.down3(m3)
        m5 = self.down4(m4)
        m = self.up1(m5, m4)
        m = self.up2(m, m3)
        m = self.up3(m, m2)  # (1,128,128,238)
        m = self.up4(m, m1)  # (1,64,256,256)
        feature_seg = m
        seg_feature = m  # （1，64，512，512）

        logits = self.outc(m)  # (1,41,512,512)

        # teeth in one ID
        # change
        # mask_values = [235]
        tooth_outputs = {}
        points_cat = []
        for tooth_pixel in mask_values:
            # tooth_pixel = tooth_pixel.item()
            # ################  test
            # tooth_pixel = 235
            # #################
            # single teeth method -创建类
            output = Single_Tooth_Grabbing(mask_original, feature_seg)
            # 调用类 image_px_mean feature map
            single_teeth = output(mask_original, feature_seg, tooth_pixel, images)
            # change
            # single_teeth = single_teeth.repeat(16, 1, 1)
            # points = (1,3,4096)  single_teeth = (1,64,4096)
            # single_point = (1,67,4096)
            single_point = torch.cat((points, single_teeth), dim=1)

            points_cat.append(single_point)
        points_cat = torch.cat(points_cat, dim=0)



        # return logits

        # 3d
        x = points_cat  # 28,67,4096

        batchsize = x.size()[0]  # 16
        n_pts = x.size()[2]  # 4096
        x, trans, trans_feat = self.feat(x)  # (16,1088,4096)
        x = F.relu(self.bn1(self.conv1(x)))  # (16,512,4096)
        x = F.relu(self.bn2(self.conv2(x)))  # (16,256,4096)
        x = F.relu(self.bn3(self.conv3(x)))  # (16,128,4096)
        x = self.conv4(x)   # (16,3,4096)
        x = x.transpose(2, 1).contiguous()  # (16,4096,3)
        # x = F.log_softmax(x.view(-1, self.k), dim=-1)  # (65536,13)
        # x = x.view(batchsize, n_pts, self.k)  # (16,4096,13)
        # tooth_tensor = {"x": x, "trans_feat": trans_feat}
        # tooth_outputs[tooth_pixel] = {"x": x, "trans_feat": trans_feat}

        # 遍历result的第0维度和mask_values列表
        for i, tooth_pixel in enumerate(mask_values):
            # 获取第 i 个 (1, 4996, 3) 数组
            x_i = x[i:i + 1]
            trans_feat_i = trans_feat[i:i + 1]
            # 将single_point添加到字典中
            tooth_outputs[tooth_pixel] = {"x": x_i, "trans_feat": trans_feat_i}

        return logits, seg_feature, tooth_outputs



class Single_Tooth_Grabbing(nn.Module):
    def __init__(self, mask_original, feature_seg):
        super(Single_Tooth_Grabbing, self).__init__()
        self.true_masks = mask_original
        self.feature_seg = feature_seg

    def forward(self, mask_original, feature_seg, tooth_pixel, images):
        # print(true_masks)        ###### go on
        bbox, tooth_pixel = get_bbox(mask_original, tooth_pixel)
        # 对全景片进行裁剪bbox
        mask_mean, pre_mask_crop, px_crop = get_px_bbox(feature_seg, bbox, mask_original, images)

        return mask_mean



class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.size_average = True
        self.reduce = True
        self.chamfer_dist = ChamferDist()
    #
    # def forward(self, pred, target, trans_feat, weight):
    #     loss = F.nll_loss(pred, target, weight=weight)
    #     mat_diff_loss = feature_transform_reguliarzer(trans_feat)
    #     total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
    #     return total_loss

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
        #
        # dist1, dist2, idx1, idx2 = self.chamfer_dist(target, pred)
        # chamfer_loss += chamfer * (torch.mean(dist1) + chamfer_opposite * torch.mean(dist2))


        ## L2 Loss
        # loss_fn = nn.MSELoss(reduction='mean')
        # loss = loss_fn(pred, target)
        #

        # Define the MSE loss function
        # MSE = nn.MSELoss(size_average=True, reduce=True)
        # device = torch.device('cuda:0')  # 数字切换卡号
        # # Calculate the MSE loss
        # loss = MSE(pred, target).to(device=device)

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




if __name__ == '__main__':
    model = get_model(13)
    points = torch.rand(16, 4096, 3)
    (model())











