import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import chamfer
from torch.autograd import Function


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        # self.k = num_class
        self.k = 3
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):  # (16,3,4096)
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
        return x, trans_feat

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
        dist1, dist2, idx1, idx2 = self.chamfer_dist(target, pred)
        chamfer_loss += chamfer * (torch.mean(dist1) + chamfer_opposite * torch.mean(dist2))


        ## L2 Loss
        # loss_fn = nn.MSELoss(reduction='mean')
        # loss = loss_fn(pred, target)
        #

        # Define the MSE loss function
        # MSE = nn.MSELoss(size_average=True, reduce=True)
        # device = torch.device('cuda:0')  # 数字切换卡号
        # # Calculate the MSE loss
        # loss = MSE(pred, target).to(device=device)
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
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))