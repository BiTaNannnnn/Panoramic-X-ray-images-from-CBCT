import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Function
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import chamfer
from unet_parts import *

###################################
#  Unet + PointNet  端到端
# pixel2mesh cat 方式 备用
###################################


class get_model(nn.Module):
    def __init__(self, n_classes, n_channels=3,  bilinear=False):
        super(get_model, self).__init__()
        # seg
        self.n_classes = n_classes
        self.n_channels = n_channels
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

        # 3d
        # self.k = num_class
        self.k = 3
        self.GPR = GraphProjection()
        self.STC = Single_Teeth_Concat()
        # self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=67)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def set_channel_size(self, channel_size):
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel_size)
        self.feat = self.feat.cuda()  # 将新的 PointNetEncoder 移动到 GPU

    def forward(self, images, mask_values, mask_original, points):  # (16,9,4096)
        #########################
        # Seg Part
        #########################
        img = images
        m = images   # (1,3,512,512)
        m1 = self.inc(m)  # (1,64,512,512)
        m2 = self.down1(m1)  # (1,128,256,256)
        m3 = self.down2(m2)  # (1,256,128,128)
        m4 = self.down3(m3)  # (1,512,64,64)
        m5 = self.down4(m4)  # (1,1024,32,32)
        m = self.up1(m5, m4)
        m = self.up2(m, m3)
        m = self.up3(m, m2)  # (1,128,128,238)
        m = self.up4(m, m1)  # (1,64,256,256)
        img_feats = [m, m2, m3, m4, m5]  # 64-126-256-512-1024
        feature_seg = m
        seg_feature = m  # （1，64，512，512）

        logits = self.outc(m)  # (1,41,512,512)

        # 原始方法拼接 （31，67，4096）
        # points_cat = self.STC(mask_values, mask_original, feature_seg, images, points)
        # pixel2mesh 方法拼接 max_size=最后一维的维度（31，1987，4096）
        points_cat, max_size = self.GPR(img_feats, points, mask_values, mask_original, images)

        #########################
        # 3D generation
        #########################
        tooth_outputs = {}
        x = points_cat  # (n牙齿数量,max_size,4096）
        channel_size = max_size
        batch_size = x.size()[0]  # 16 batchsize= 牙齿的个数
        n_pts = x.size()[2]  # 4096

        # 设置channel_size
        self.set_channel_size(channel_size)
        x, trans, trans_feat = self.feat(x)  # (16,1088,4096)
        # x, max_size = self.GPR(img_feats, x, mask_values, mask_original, images)
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


class Single_Teeth_Concat(nn.Module):

    def __init__(self):
        super(Single_Teeth_Concat, self).__init__()

    def forward(self, mask_values, mask_original, feature_seg, images, points):
        # teeth in one ID
        # change
        # mask_values = [235]

        points_cat = {}
        for tooth_pixel in mask_values:
            # tooth_pixel = tooth_pixel.item()
            # ################  test
            # tooth_pixel = 235
            # #################
            # single teeth method -输出output=可做拼接的feature points,pre_mask_crop=剪裁好的predict mask,px crop=剪裁过的px
            output = Single_Tooth_Grabbing(mask_original, feature_seg, images)
            # 调用类 image_px_mean feature map
            # pre_mask_crop = 预测的mask crop，px_crop = 全景片单颗牙齿crop
            mask_mean, tooth_pixel, pre_mask_crop, px_crop = output(mask_original, feature_seg, tooth_pixel, images)
            # change
            # single_teeth = single_teeth.repeat(16, 1, 1)
            # points = (1,3,4096)  single_teeth = (1,64,4096)
            # single_point = (1,67,4096)
            single_point = torch.cat((points, mask_mean), dim=1)  # (1,3,4096)+(1,64,4096) = (1,67,4096)
            points_cat[tooth_pixel] = {"ID": tooth_pixel, "single_points": single_point}
        points_cat = {item['ID']: item['single_points'] for item in points_cat.values()}
        points_cat = torch.cat(list(points_cat.values()), dim=0)
        # (1,67,4096)-->(31,67,4096)

        return points_cat

class Single_Tooth_Grabbing(nn.Module):
    def __init__(self, mask_original, feature_seg, images):
        super(Single_Tooth_Grabbing, self).__init__()
        self.true_masks = mask_original
        self.feature_seg = feature_seg
        self.images = images

    def forward(self, mask_original, feature_seg, tooth_pixel, images):
        # print(true_masks)        ###### go on
        bbox, tooth_pixel = get_bbox(mask_original, tooth_pixel)
        # 对全景片进行裁剪bbox
        mask_mean, pre_mask_crop, px_crop = get_px_bbox(feature_seg, bbox, mask_original, images)

        return mask_mean, tooth_pixel, pre_mask_crop, px_crop


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


class GraphProjection(nn.Module):
    """Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self):
        super(GraphProjection, self).__init__()

    def forward(self, img_features, input, mask_values, mask_original, images):
        self.img_feats = img_features
        points_cat = {}
        for tooth_pixel in mask_values:
            # tooth_pixel = tooth_pixel.item()
            # ################  test
            # tooth_pixel = 235
            # #################

            # get 3D 坐标
            h = input[:, 0, :]  # 获取x坐标
            w = input[:, 1, :]  # 获取y坐标
            h = h.squeeze()
            w = w.squeeze()
            # h = torch.clamp(h, min=0, max=223)
            # w = torch.clamp(w, min=0, max=223)

            # get 2D single teeth box
            box_con, tooth_pixel = get_bbox(mask_original, tooth_pixel)
            x_min = max(0, np.min(box_con[:, 0]))
            y_min = max(0, np.min(box_con[:, 1]))
            x_max = np.max(box_con[:, 0])
            y_max = np.max(box_con[:, 1])
            images_crop = images[:, :, y_min:y_max, x_min:x_max]  # 单个牙齿的img
            if images_crop.shape[-1] == 0 or images_crop.shape[-2] == 0:
                print("The last two dimensions of img_feat should not be zero.")
            _, _, img_h, img_w = images_crop.shape  # eg （1，3，86，24）
            # if img_h >= 160 or img_w >= 160:
            #     print("The img_feat.shape -img_h  should <128.")
            # 256 大小
            h_256 = img_h / 2
            w_256 = img_w / 2
            # 128 大小
            h_128 = h_256 / 2
            w_128 = w_256 / 2
            # 64 大小
            h_64 = h_128 / 2
            w_64 = w_128 / 2
            # 32 大小
            h_32 = h_64 / 2
            w_32 = w_64 / 2

            # img_sizes = [512, 256, 128, 64, 32]
            # img_sizes 是每个牙齿的crop之后的尺寸
            img_sizes = [(max(img_h, 1), max(img_w, 1)),  # 512 每个值最小=1
                         (max(img_h, 1), max(img_w, 1)),
                         (max(h_256, 1), max(w_256, 1)),
                         (max(h_128, 1), max(w_128, 1)),
                         (max(h_64, 1), max(w_64, 1)),
                         (max(h_32, 1), max(w_32, 1))]
            feats = []
            for i in range(6):
                img_feat_crop = Img_Feat_Crop(self, i, images_crop, x_min, y_min, x_max, y_max)
                out = self.project(i, h, w, img_sizes[i], img_feat_crop)
                feats.append(out)

            output = torch.cat(feats, 1)  # (1,1987，4096)
            max_size = output.size(1)
            points_cat[tooth_pixel] = {"ID": tooth_pixel, "points_cat": output}
        points_cat = {item['ID']: item['points_cat'] for item in points_cat.values()}
        points_cat = torch.cat(list(points_cat.values()), dim=0)
        # 用于 size不匹配的padding 对称填充
        # # 找到最大的维度大小
        # max_size = max(tensor.size(-1) for tensor in points_cat_list)
        # # 对每个张量进行填充，使其在最后一个维度上达到 max_size，且填充是两边对称的
        # points_cat_list = [F.pad(tensor, ((max_size - tensor.size(-1)) // 2,
        #                                   (max_size - tensor.size(-1) + 1) // 2)) for tensor in points_cat_list]
        # # 沿着第一维度拼接
        # points_cat = torch.cat(points_cat_list, dim=0)
        # # (n牙齿数量,max_size,4096)
        # points_cat = points_cat.permute(0, 2, 1)

        return points_cat, max_size

    def project(self, index, h, w, img_size, img_feat_crop):
        img_feat = img_feat_crop
        x = h * 199.
        y = w * 199.
        # h_list = h.tolist()
        # print(h_list)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()
        x1 = torch.clamp(x1, max=199)
        y1 = torch.clamp(y1, max=199)
        x2 = torch.clamp(x2, max=199)
        y2 = torch.clamp(y2, max=199)

        # img_feat=(1,3,86,24) Q11=(1,3,1,4096)
        # print('img_feat  = ', img_feat.shape)
        # Xmax_value = x.max()
        # Ymax_value = y.
        # print('x  = ', Xmax_value)
        # print('y  = ', Ymax_value)
        Q11 = img_feat[:, :, x1.long(), y1.long()].clone()
        Q12 = img_feat[:, :, x1.long(), y2.long()].clone()
        Q21 = img_feat[:, :, x2.long(), y1.long()].clone()
        Q22 = img_feat[:, :, x2.long(), y2.long()].clone()

        x, y = x.long(), y.long()

        weights = torch.mul(x2 - x, y2 - y)  # (1,4096) --(31,4096)
        weights = cheack_tensor(weights, Q11)  # (1,31，1，4096，)
        # print('weights tensor = ', weights.float().view(-1, 1).shape)  # (4096,1) (4096,31)
        # print('Q11 tensor = ', torch.transpose(Q11, 0, 1).shape)  # (1,1,4096,24)
        # Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))  #(1,1,4096,24)-- (1,31,4096,24)
        Q11 = torch.mul(weights.float(), Q11)  # (1,31，1,4096)

        weights = torch.mul(x2 - x, y - y1)
        weights = cheack_tensor(weights, Q12)
        Q12 = torch.mul(weights.float(), Q12)

        weights = torch.mul(x - x1, y2 - y)
        weights = cheack_tensor(weights, Q21)
        Q21 = torch.mul(weights.float(), Q21)

        weights = torch.mul(x - x1, y - y1)
        weights = cheack_tensor(weights, Q22)
        Q22 = torch.mul(weights.float(), Q22)

        output = Q11 + Q21 + Q12 + Q22  # (1,3,4096)
        output = output.squeeze(2)  # （1，3，4096）

        return output

def cheack_tensor(weights, Q11):
    # 获取 weights 和 Q11 的形状
    weights_shape = weights.shape
    Q11_shape = Q11.shape
    # 检查 weights 和 Q11 的第一维度是否相同
    if weights_shape != Q11_shape:
        # 如果不同，则在 weights = Q11维度
        weights = weights.unsqueeze(0).unsqueeze(1)
        weights = weights.expand_as(Q11)
    return weights


def Img_Feat_Crop(self, i, images_crop, x_min, y_min, x_max, y_max):
    crop_h = images_crop.shape[2]
    crop_w = images_crop.shape[3]
    if i == 0:  # 原始大小 channel = 3
        img_feat_crop = images_crop
    elif i == 1:  # 原始大小 channel = 64
        x_min64 = int(x_min)
        y_min64 = int(y_min)
        x_max64 = int(x_max)
        y_max64 = int(y_max)
        img_feat_crop = self.img_feats[0][:, :, y_min64:y_max64, x_min64:x_max64]
        width_new = x_max64 - x_min64
        height_new = y_max64 - y_min64
    elif i == 2:  # 128
        x_min128 = int(x_min / 2)
        y_min128 = int(y_min / 2)
        x_max128 = max(x_min128 + 1, int(x_max / 2))
        y_max128 = max(y_min128 + 1, int(y_max / 2))
        img_feat_crop = self.img_feats[1][:, :, y_min128:y_max128, x_min128:x_max128]
        #  差值成他原本大小
        img_feat_crop = F.interpolate(img_feat_crop, size=(crop_h, crop_w), mode='nearest')
        width_new = x_max128 - x_min128
        height_new = y_max128 - y_min128
    elif i == 3:  # 256
        x_min256 = int(x_min / 4)
        y_min256 = int(y_min / 4)
        x_max256 = max(x_min256 + 1, int(x_max / 4))
        y_max256 = max(y_min256 + 1, int(y_max / 4))
        img_feat_crop = self.img_feats[2][:, :, y_min256:y_max256, x_min256:x_max256]
        img_feat_crop = F.interpolate(img_feat_crop, size=(crop_h, crop_w), mode='nearest')
        width_new = x_max256 - x_min256
        height_new = y_max256 - y_min256
    elif i == 4:  # 512
        x_min512 = int(x_min / 8)
        y_min512 = int(y_min / 8)
        x_max512 = max(x_min512 + 1, int(x_max / 8))
        y_max512 = max(y_min512 + 1, int(y_max / 8))
        img_feat_crop = self.img_feats[3][:, :, y_min512:y_max512, x_min512:x_max512]
        img_feat_crop = F.interpolate(img_feat_crop, size=(crop_h, crop_w), mode='nearest')
        width_new = x_max512 - x_min512
        height_new = y_max512 - y_min512
    elif i == 5:  # 1024
        x_min1024 = int(x_min / 16)
        y_min1024 = int(y_min / 16)
        #  确保裁剪出来的 img_feat_crop 的宽度和高度都不为零
        x_max1024 = max(x_min1024 + 1, int(x_max / 16))
        y_max1024 = max(y_min1024 + 1, int(y_max / 16))
        img_feat_crop = self.img_feats[4][:, :, y_min1024:y_max1024, x_min1024:x_max1024]
        img_feat_crop = F.interpolate(img_feat_crop, size=(crop_h, crop_w), mode='nearest')
        width_new = x_max1024 - x_min1024
        height_new = y_max1024 - y_min1024

    # 计算在高度和宽度上需要填充的大小 向上取整 （w=200,h=200）
    pad_height_up = math.floor((200 - crop_h) / 2)
    pad_height_down = math.ceil((200 - crop_h) / 2)
    pad_width_left = math.floor((200 - crop_w) / 2)
    pad_width_right = math.ceil((200 - crop_w) / 2)

    # 使用 padding 函数进行填充 格式为 (左, 右, 上, 下)
    feat_crop_padded = F.pad(img_feat_crop, (pad_width_left, pad_width_right, pad_height_up, pad_height_down))

    return feat_crop_padded


if __name__ == '__main__':
    model = get_model(13)
    points = torch.rand(16, 4096, 3)
    (model())