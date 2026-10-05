import re

import torch
from pyemd import emd_samples
import numpy as np
from scipy.stats import entropy
import torch.nn.functional as F
from numpy.linalg import norm
import trimesh
import os
import open3d as o3d
from torch.autograd import Function

# def compute_cov_cd_emd(sample_pcs, ref_pcs, batch_size=1, accelerated_cd=False, reduced=True, threshold=0.1):
#     N_sample = sample_pcs.shape[1]
#     ref_pcs = ref_pcs.unsqueeze(0)
#     N_ref = ref_pcs.shape[1]
#     assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)
#
#     cd_lst = []
#     emd_lst = []
#     iterator = range(0, N_sample, batch_size)
#
#     for b_start in iterator:
#         b_end = min(N_sample, b_start + batch_size)
#         sample_batch = sample_pcs[b_start:b_end]
#         ref_batch = ref_pcs[b_start:b_end]
#
#         if accelerated_cd:
#             dl, dr = distChamfer(sample_batch, ref_batch)
#         else:
#             dl, dr = distChamfer(sample_batch, ref_batch)
#         cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))
#
#         emd_batch = emd_approx(sample_batch, ref_batch)
#         emd_lst.append(emd_batch)
#
#     if reduced:
#         cd = torch.cat(cd_lst).mean()
#         emd = torch.cat(emd_lst).mean()
#     else:
#         cd = torch.cat(cd_lst)
#         emd = torch.cat(emd_lst)
#
#     # 计算 COV-CD
#     coverage_cd = (cd < threshold).float().mean()
#     cov_cd = cd * coverage_cd
#
#     # 计算 COV-EMD
#     coverage_emd = (emd < threshold).float().mean()
#     cov_emd = emd * coverage_emd
#
#     # results = {
#     #     'MMD-CD': cd,
#     #     'MMD-EMD': emd,
#     #     'COV-CD': cov_cd,
#     #     'COV-EMD': cov_emd,
#     # }
#     return cd, emd, cov_cd,  cov_emd,

def compute_metrics(sample_pcs, ref_pcs, accelerated_cd=False, threshold=1.0):
    ref_pcs = ref_pcs.unsqueeze(0)
    # 计算 Chamfer 距离
    if accelerated_cd:
        dl, dr = distChamfer(sample_pcs, ref_pcs)
    else:
        dl, dr = distChamfer(sample_pcs, ref_pcs)

    cd = dl.mean() + dr.mean()

    # 计算 Earth Mover's Distance
    emd_batch = emd_approx(sample_pcs, ref_pcs)
    emd = emd_batch.mean()

    # 计算 COV-CD
    coverage_cd = (cd < threshold).float().mean()
    cov_cd = cd * coverage_cd

    # 计算 COV-EMD
    coverage_emd = (emd < threshold).float().mean()
    cov_emd = emd * coverage_emd

    # results = {
    #     'MMD-CD': cd,
    #     'MMD-EMD': emd,
    #     'COV-CD': cov_cd,
    #     'COV-EMD': cov_emd,
    # }
    return cd, emd, cov_cd,  cov_emd


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def emd_approx(sample_batch, ref_batch):
    """
    计算 Earth Mover's 距离的近似值
    :param sample_batch: 样本点云 (batch_size, num_points, num_dimensions)
    :param ref_batch: 参考点云 (batch_size, num_points, num_dimensions)
    :return: Earth Mover's 距离的近似值
    """
    batch_size, num_points, _ = sample_batch.shape

    emd_distances = []

    for i in range(batch_size):
        # 将三维数组展平为一维数组
        sample_flat = sample_batch[i].cpu().flatten().numpy()
        ref_flat = ref_batch[i].cpu().flatten().numpy()

        # 计算 Earth Mover's Distance
        emd_distance = emd_samples(sample_flat, ref_flat)
        emd_distances.append(torch.tensor(emd_distance, dtype=torch.float32))  # 返回 PyTorch 张量

    return torch.stack(emd_distances)



if __name__ == '__main__':

    mesh_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT/'
    point_path= 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/visual/0117_CrossAttention_95_Infer/'
    # # 获取文件夹a和文件夹b中的所有文件夹名
    # mesh_file = os.listdir(mesh_path)
    # point_file= os.listdir(point_path)
    #
    # # GT_points = read_teeth_mesh(mesh_path)
    #
    # # 遍历文件夹a中的每个文件夹
    # for point_file in point_file:
    #     ID = point_file
    #     meshID_path = os.path.join(mesh_path, ID)
    #     pointID_path = os.path.join(point_path, ID)
    #     # 获取文件夹a和文件夹b中的所有文件夹名
    #     pointID_path_list = os.listdir(pointID_path)
    #     # 遍历文件夹a中的每个文件夹
    #     for pointID in pointID_path_list:
    #         if pointID.endswith(".obj"):
    #             point_teeth = o3d.io.read_point_cloud(pointID_path + pointID)
    #             point_cloud = point_cloud.uniform_down_sample(every_k_points=int(len(point_cloud.points) / 4096))
    #             # 获取匹配的文件夹名
    #             point_teeth_id = pointID.split('_')[1].split('.')[0]
    #             mesh_teeth_name = f"{int(int(point_teeth_id) / 5)}._Root.stl"
    #             mesh_teeth_path = os.path.join(meshID_path, mesh_teeth_name)
    #             mesh_teeth = read_teeth_mesh(mesh_teeth_path)
    #             ''' 计算 '''
    #             result = compute_cov_cd_emd(point_teeth, mesh_teeth)


















