'''
0117_CrossAttention_all = attention 放在输出之前
解决：你用的不对，
1. 你的attention是把点云的特征融合到图像上，我们要做的是把图像的特征融合到点云上；
2. img feature 不要用uNet 里面的x5 feat --> 该用每颗牙齿的feat cat 起来 bs = point bs
同时修改了unet_parts.py

ps 之前效果都不好 放在了output 之前 ，所以这次修改到第一个conv 之后

修改了img 的没必要的维度更改

最终版本
'''


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
# from fightingcv_attention.attention.PSA import *
from fightingcv_attention.attention.PolarizedSelfAttention import SequentialPolarizedSelfAttention
from fightingcv_attention.attention.CrissCrossAttention import CrissCrossAttention

from torch.autograd import Function
from torch.nn import init
import torch.nn.functional as F
import pytorch3d
import os
import sys
sys.path.append(os.path.dirname(__file__))
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
        self.unet_part = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
        self.feature_part = Feature_Net(n_channels=3, n_classes=n_classes, bilinear=bilinear)
        self.pointnet_part = PointNet_3d()


    def forward(self, images, mask_values, mask_original, points):  # (16,9,4096)
        '''seg part'''
        m = images
        logits, seg_feature = self.unet_part(m)
        '''Featrure Part'''
        logist_3d, feature_seg, x5_feat = self.feature_part(m)
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
        single_teeth_cat = torch.cat(single_teeth_cat, dim=0)  # bs， 64，80,80

        '''3d part'''
        x = points_cat  # 28,67,4096
        tooth_outputs = self.pointnet_part(x, mask_values, tooth_outputs, single_teeth_cat)

        return logits, seg_feature, tooth_outputs


class Feature_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        # bilinear = 是否采用双线性差值
        super(Feature_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.psa_x5 = PSA(channel=1024, reduction=8)
        self.psa_x4 = PSA(channel=512, reduction=8)
        self.psa_x = PSA(channel=512, reduction=8)
        # paper:"Polarized Self-Attention: Towards High-quality Pixel-wise Regression"
        self.SPSA = SequentialPolarizedSelfAttention(channel=512)

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
        '''Feature_Net'''
        x1 = self.inc(x)  # (1,64,512,622)
        x2 = self.down1(x1)  # (1,128,256,311)
        x3 = self.down2(x2)  # (1,256,128,155)
        x4 = self.down3(x3)  # (1,512,64,77)
        x5 = self.down4(x4)  # (1,1024,32,38)
        # x = self.SPSA(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature_seg = x
        logits_3d = self.outc(x)
        return logits_3d, feature_seg, x5


class PointNet_3d(nn.Module):
    def __init__(self):
        super(PointNet_3d, self).__init__()
        # self.k = num_class
        self.k = 3
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=67)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        # self.conv4 = torch.nn.Conv1d(8, self.k, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.cross_attention_module = CrossAttention(input_dim_img=6400, input_dim_points=4096, output_dim=1024, num_heads=8)


    def forward(self, x, mask_values, tooth_outputs, single_teeth_cat):  # (16,3,4096)
        tooth_num = x.size()[0]  # 16
        n_pts = x.size()[2]  # 4096
        x, trans, trans_feat = self.feat(x)  # (16,1088,4096)

        output_features = self.cross_attention_module(single_teeth_cat, x)

        x = F.relu(self.bn1(self.conv1(output_features)))  # (16,512,4096)
        x = F.relu(self.bn2(self.conv2(x)))  # (16,256,4096)
        x = F.relu(self.bn3(self.conv3(x)))  # (16,128,4096)
        x = self.conv4(x)  # (tooth sum,3,4096)
        x = x.transpose(2, 1).contiguous()  # (16,4096,3)
        # 遍历result的第0维度和mask_values列表
        for i, tooth_pixel in enumerate(mask_values):
            # 获取第 i 个 (1, 4996, 3) 数组
            x_i = x[i:i + 1]
            trans_feat_i = trans_feat[i:i + 1]
            # 将single_point添加到字典中
            tooth_outputs[tooth_pixel] = {"x": x_i, "trans_feat": trans_feat_i}

        return tooth_outputs


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
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
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



class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super(SequentialPolarizedSelfAttention, self).__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out


class PSA(nn.Module):
    def __init__(self, channel=512, reduction=4, S=4):
        super(PSA, self).__init__()
        self.S = S
        self.convs = nn.ModuleList([nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1) for i in range(S)])
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for i in range(S)
        ])

        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        # Step1:SPC module
        # Split the input tensor into S parts along the channel dimension
        SPC_out_split = torch.split(x, c // self.S, dim=1)
        SPC_out_conv = [conv(spc) for spc, conv in zip(SPC_out_split, self.convs)]
        SPC_out = torch.cat(SPC_out_conv, dim=1)

        # Reshape for SE blocks
        SPC_out_reshaped = SPC_out.view(b, self.S, c // self.S, h, w)

        # Step2:SE weight
        SE_out = [se(SPC_out_reshaped[:, idx, :, :, :]) for idx, se in enumerate(self.se_blocks)]
        SE_out_stacked = torch.stack(SE_out, dim=1)
        SE_out_expanded = SE_out_stacked.expand_as(SPC_out_reshaped)

        # Step3:Softmax
        softmax_out = self.softmax(SE_out_expanded)

        # Step4:SPA
        PSA_out = SPC_out_reshaped * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)

        return PSA_out

'''Point attention'''

class CrossAttention(nn.Module):
    def __init__(self, input_dim_img, input_dim_points, output_dim, num_heads):
        super(CrossAttention, self).__init__()

        self.query_projection = nn.Linear(input_dim_points, output_dim)
        self.key_projection = nn.Linear(input_dim_img, output_dim)
        self.value_projection = nn.Linear(input_dim_img, output_dim)
        self.conv1x1 = nn.Conv1d(64, 1088, kernel_size=1)
        self.multihead_attention = nn.MultiheadAttention(1024, num_heads)

    def forward(self, image_features, point_cloud_features):
        B, C, W, H = image_features.size() # bs，64，80，80
        B_P, C_P, N = point_cloud_features.size()  # bs,1088,4096  // C=3
        # Reshape image features to (B, C, W*H)
        image_features = image_features.view(B, C, -1)  # bs 64 6400
        # image_features = self.conv1x1(image_features)  # bs，3，6400
        # image_features = image_features.expand(B_P, C_P, -1)  # bs，3，6400
        '''
        imput_img_dim = image_features (B, C, -1)最后一维 = 6400
        imput_point_dim = point_cloud_features 最后一维= 4096
        '''
        # Project queries, keys, and values
        query = self.query_projection(point_cloud_features)  # bs，3，1024
        key = self.key_projection(image_features)  # bs，64，1024
        value = self.value_projection(image_features)  # bs,64,1024

        # Transpose to (seq_len, batch, features) for MultiheadAttention
        query = query.permute(1, 0, 2)  # (1088, bs， 1024)
        key = key.permute(1, 0, 2)  # 64，bs，1024
        value = value.permute(1, 0, 2)  # 64，bs，1024

        # Multihead attention
        attn_output, _ = self.multihead_attention(query, key, value)  # 1088，bs, 1024

        # Transpose back to (batch, features, seq_len)
        attn_output = attn_output.permute(1, 0, 2)  # 26，1088,1024

        # Reshape back to (B, C, N)
        attn_output = F.interpolate(attn_output, size=(4096,), mode='nearest')

        return attn_output




# Point-wise attention for each voxel
class PACALayer(nn.Module):
    def __init__(self, dim_ca, dim_pa, reduction_r):
        super(PACALayer, self).__init__()
        self.pa = PALayer()
        self.ca = CALayer()
        self.sig = nn.Sigmoid()

    def forward(self, x):  # 形状为 (num, 3, 4096)
        pa_weight = self.pa(x)  #（1，bs，4096）
        ca_weight = self.ca(x)  #（1,bs,3）
        pa_weight = pa_weight.unsqueeze(-1)  # 形状变为 (1, bs, 4096, 1)
        ca_weight = ca_weight.unsqueeze(2)  # 形状变为 (1, bs, 1, 3)
        paca_weight = torch.mul(pa_weight, ca_weight)  # (1, bs, 4096, 3)
        paca_weight = paca_weight.squeeze().permute(0, 2, 1)
        paca_normal_weight = self.sig(paca_weight)
        out = torch.mul(x, paca_normal_weight)
        return out, paca_normal_weight


# Point-wise attention for each voxel
class PALayer(nn.Module):
    def __init__(self):
        super(PALayer, self).__init__()
        dim_pa = 4096
        reduction_pa = 16
        self.fc = nn.Sequential(
            nn.Linear(dim_pa, dim_pa // reduction_pa),
            nn.ReLU(inplace=True),
            nn.Linear(dim_pa // reduction_pa, dim_pa)
        )

    def forward(self, x):
        b, w, n = x.size()
        y, _ = torch.max(x, dim=1, keepdim=True)
        y = y.permute(1, 0, 2)
        y = y[0].view(b, n)  # （bs，4096）
        out1 = self.fc(y)   # （bs，4096）
        out1 = out1.view(1, b, n)  # （1，bs，4096）
        return out1

# Channel-wise attention for each voxel
class CALayer(nn.Module):
    def __init__(self):
        super(CALayer, self).__init__()
        dim_ca = 3
        reduction_ca = 1
        self.fc = nn.Sequential(
            nn.Linear(dim_ca, dim_ca // reduction_ca),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ca // reduction_ca, dim_ca)
        )

    def forward(self, x):
        b, c, n = x.size()
        y = torch.max(x, dim=2, keepdim=True)[0].view(b, c)  #（bs,3）
        y = self.fc(y)  #（bs,3）
        y = y.view(1, b,  c)  #（1,bs,3）
        return y


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)


        return x1


#
# class get_loss(nn.Module):
#     def __init__(self, mat_diff_loss_scale=1, emd_loss_scale=1):
#         super(get_loss, self).__init__()
#         self.mat_diff_loss_scale = mat_diff_loss_scale
#         self.emd_loss_scale = emd_loss_scale
#         self.chamfer_dist = ChamferDist()
#
#     def forward(self, pred, target):
#         # Chamfer Loss
#         dist1, dist2, _, _ = self.chamfer_dist(target, pred)
#         chamfer_loss = torch.mean(dist1) + torch.mean(dist2)
#
#         # EMD Loss
#         emd_loss = self.emd_loss(pred, target)
#
#         # Combine the losses
#         total_loss = self.mat_diff_loss_scale * chamfer_loss + self.emd_loss_scale * emd_loss
#         return total_loss
#
#     def emd_loss(self, pred, target, eps=1e-5):
#         batch_size, num_points, _ = pred.shape
#         emd_loss = 0.0
#
#         # 将循环内的操作移动到GPU
#         for i in range(batch_size):
#             cost_matrix = torch.cdist(pred[i], target[i], p=2)
#             # 使用库函数进行线性求和分配，仍然在CPU上进行
#             row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
#             matched_distances = cost_matrix[row_ind, col_ind]
#             emd_loss += torch.mean(matched_distances)
#
#         emd_loss /= batch_size
#         return emd_loss


'''原始charmfor loss'''
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




if __name__ == '__main__':
    model = get_model(13)
    points = torch.rand(16, 4096, 3)
    (model())