from PIL import Image
import numpy as np
import trimesh
import time
import torch
from PIL import ImageEnhance, Image
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
from torch.utils.data import Dataset
from skimage import io, transform
from os.path import splitext, isfile, join
from scipy.ndimage import zoom
from os import listdir
from torchvision.transforms import ToTensor
from pathlib import Path
import os
import pyvista as pv

# 记录读取数据之前的时间
start_time = time.time()
class ToothLoader(Dataset):
    def __init__(self, dir_img: str, dir_mask: str, dir_GT: str,
                num_point=4096, img_scale=1.0,  block_size=1.0, transform=None, mask_suffix: str = '_mask'):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        assert 0 < img_scale <= 1, 'Scale must be between 0 and 1'
        self.scale = img_scale
        self.mask_suffix = mask_suffix
        # dir_img = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_test1/imgs/' # Data_Tooth
        # dir_mask = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_test1/masks/'
        # dir_GT = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT1/'
        self.images_dir = Path(dir_img)
        self.mask_dir = Path(dir_mask)
        self.Gt_dir = Path(dir_GT)
        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if isfile(join(self.images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')

        print(f'Creating dataset with {len(self.ids)} examples')
        print('Scanning mask files to determine unique values')
        unique = []
        for idx in self.ids:
            result = unique_mask_values(idx, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix)
            unique.append(result)

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        print(f'Unique mask values: {self.mask_values}')


    def __getitem__(self, idx):
        # start_time_data = time.time()
        ################################ seg process
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        mesh_path = os.path.join(self.Gt_dir, name)
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert mesh_path, f'Either no mesh_file or multiple masks found for the ID {name}: {mesh_path}'
        # mask_original process
        # mask = [0-28]
        # mask_original =0,55,60...235]
        mask = load_image(mask_file[0])
        mask_original = mask.copy()

        mask_original = np.asarray(mask_original)
        unique_values = np.unique(mask)
        # print(name)
        # print(f'Unique mask values: {unique_values}')
        non_zero_values = unique_values[unique_values != 0]
        unique_values = list(sorted(non_zero_values.tolist()))
        # ignore impacted tooth
        unique_values = [x for x in unique_values if x >= 50]
        values_to_remove = [240, 190, 140, 90]  # 需要移除的值 保留28颗牙齿
        for value in values_to_remove:
            while value in unique_values:
                unique_values.remove(value)
        # print(f'Unique mask values: {发现unique_values}')

        # img process
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # '''增加img 颜色抖动'''
        # img = randomColor(img)
        # plt.imshow(img)  # 使用合适的颜色映射，对于灰度图像，通常使用 'gray'
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        #
        # plt.imshow(mask, cmap='gray')  # 使用合适的颜色映射，对于灰度图像，通常使用 'gray'
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()

        '''发现unique_values（某个id所有牙齿）；self.mask_values （47+1个分类）作用不一致'''
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        # print(img.shape)
        # print(mask.shape)
        # print(np.unique(mask))
        # exit()



        ################################ 3d process

        # 读入一个id 的牙齿mesh
        GT_points = read_teeth_mesh(mesh_path)

        ################################ 3d process  -- GET POINTS

        # # points 方法一 = 正方体均匀取点
        # n = 16
        # x = torch.linspace(0, 1, n)
        # y = torch.linspace(0, 1, n)
        # z = torch.linspace(0, 1, n)
        # xx, yy, zz = torch.meshgrid(x, y, z)
        # points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
        # # # 验证
        # with open('xxx/Project/' + str(int(1)) + '.obj', 'w') as f:
        #     for point in points:
        #         f.write(f'v {point[0]} {point[1]} {point[2]}\n')

        # points 方法二 = 正方体随机取点
        # points = torch.rand(1, 4096, 3)

        # points 方法三 = 球表面积均匀取点
        # points = fibonacci_sphere(samples=4096)

        # # 计算并打印读取数据所需的时间
        # print("Time taken to read one data: ", time.time() - start_time_data)

        return {
            'ID': name,
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            # 'points': points,
            'GT_points': GT_points,
            'mask_values': unique_values,
            'mask_original': torch.as_tensor(mask_original.copy()).long().contiguous(),
        }



    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img


class ToothLoader_voxel(Dataset):
    def __init__(self, dir_img: str, dir_mask: str, dir_GT: str,
                num_point=4096, img_scale=1.0,  block_size=1.0, transform=None, mask_suffix: str = '_mask'):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        assert 0 < img_scale <= 1, 'Scale must be between 0 and 1'
        self.scale = img_scale
        self.mask_suffix = mask_suffix
        # dir_img = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_test1/imgs/' # Data_Tooth
        # dir_mask = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_test1/masks/'
        # dir_GT = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT1/'
        self.images_dir = Path(dir_img)
        self.mask_dir = Path(dir_mask)
        self.Gt_dir = Path(dir_GT)
        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if isfile(join(self.images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')

        print(f'Creating dataset with {len(self.ids)} examples')
        print('Scanning mask files to determine unique values')
        unique = []
        for idx in self.ids:
            result = unique_mask_values(idx, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix)
            unique.append(result)

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        print(f'Unique mask values: {self.mask_values}')


    def __getitem__(self, idx):
        # start_time_data = time.time()
        ################################ seg process
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        mesh_path = os.path.join(self.Gt_dir, name)
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert mesh_path, f'Either no mesh_file or multiple masks found for the ID {name}: {mesh_path}'
        # mask_original process
        # mask = [0-28]
        # mask_original =0,55,60...235]
        mask = load_image(mask_file[0])
        mask_original = mask.copy()

        mask_original = np.asarray(mask_original)
        unique_values = np.unique(mask)
        # print(name)
        # print(f'Unique mask values: {unique_values}')
        non_zero_values = unique_values[unique_values != 0]
        unique_values = list(sorted(non_zero_values.tolist()))
        # ignore impacted tooth
        unique_values = [x for x in unique_values if x >= 50]
        values_to_remove = [240, 190, 140, 90]  # 需要移除的值 保留28颗牙齿
        for value in values_to_remove:
            while value in unique_values:
                unique_values.remove(value)
        # print(f'Unique mask values: {发现unique_values}')

        # img process
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # '''增加img 颜色抖动'''
        # img = randomColor(img)
        # plt.imshow(img)  # 使用合适的颜色映射，对于灰度图像，通常使用 'gray'
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        #
        # plt.imshow(mask, cmap='gray')  # 使用合适的颜色映射，对于灰度图像，通常使用 'gray'
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()

        '''发现unique_values（某个id所有牙齿）；self.mask_values （47+1个分类）作用不一致'''
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        # print(img.shape)
        # print(mask.shape)
        # print(np.unique(mask))
        # exit()



        ################################ 3d process

        # 读入一个id 的牙齿mesh
        # GT_points = read_teeth_mesh(mesh_path)

        # 读入一个id 的牙齿voxel
        GT_voxel = read_teeth_mesh_voxel(mesh_path)

        ################################ 3d process  -- GET POINTS

        # # points 方法一 = 正方体均匀取点
        # n = 16
        # x = torch.linspace(0, 1, n)
        # y = torch.linspace(0, 1, n)
        # z = torch.linspace(0, 1, n)
        # xx, yy, zz = torch.meshgrid(x, y, z)
        # points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
        # # 验证
        # with open('xxx/Project/' + str(int(1)) + '.obj', 'w') as f:
        #     for point in points:
        #         f.write(f'v {point[0]} {point[1]} {point[2]}\n')

        # points 方法二 = 正方体随机取点
        # points = torch.rand(1, 4096, 3)

        # points 方法三 = 球表面积均匀取点
        # points = fibonacci_sphere(samples=4096)

        # # 计算并打印读取数据所需的时间
        # print("Time taken to read one data: ", time.time() - start_time_data)

        return {
            'ID': name,
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            # 'points': points,
            'GT_points': GT_voxel,
            'mask_values': unique_values,
            'mask_original': torch.as_tensor(mask_original.copy()).long().contiguous(),
        }

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img


def randomColor(image):
    '''调整img 颜色抖动'''
    # plt.imshow(image)
    # plt.show()
    # 颜色饱和度
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    # 亮度
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    # 对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))
    # 显示增强后的图片
    # plt.imshow(image)
    # plt.show()
    return image

def read_teeth_mesh_voxel(mesh_path, pitch=0.1):
    Person_GT = {}
    ID = mesh_path.split('/')[-1]
    for subdir, _, files in os.walk(mesh_path):
        files = sorted(files)
        for file in files:
            if file.endswith('.bin') and '._' in file:
                file_num = file.split('._')[0]
                file_name = int(file_num) * 5
                file_path = os.path.join(subdir, file)

                voxel_size = [11, 11, 11]

                # Assuming the binary voxel data is stored as uint8
                with open(file_path, 'rb') as file:
                    voxel_data = np.fromfile(file, dtype=np.uint8)

                    # Reshape the voxel data to the desired shape (assuming it's 64x64x64)
                    voxel_data = voxel_data.reshape(voxel_size)

                    # If you want to resize the voxel data, you can use zoom
                    zoom_factor = [voxel_size[i] / voxel_size[i] for i in range(3)]
                    voxel_data = zoom(voxel_data, zoom_factor)

                    # # Create a trimesh object
                    # voxel_mesh = trimesh.voxel.VoxelGrid(voxel_data)
                    #
                    # # Save voxel data as an OBJ file
                    # save_obj_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT_bin2obj/' + ID + '/'+ file_num + '._Root.obj'
                    # voxel_mesh.export(save_obj_path)

            Person_GT[file_name] = voxel_data
    return Person_GT
# def read_teeth_mesh_voxel(mesh_path, pitch=0.1):
#     Person_GT = {}
#     for subdir, _, files in os.walk(mesh_path):
#         files = sorted(files)
#         for file in files:
#             if file.endswith('.obj') and '._' in file:
#                 file_num = file.split('._')[0]
#                 file_name = int(file_num) * 5
#                 file_path = os.path.join(subdir, file)
#                 voxel_size = [64, 64, 64]
#                 voxel = pv.read(file_path)
#                 # Voxelize the mesh
#                 voxel = voxel.voxelize(voxel_size)
#                 voxel_data = voxel.point_arrays['ImageScalars']
#
#
#                 # 读取voxel
#                 with open(file_path, 'rb') as file:
#                     new_shape = 786432
#                     # Assuming the voxel data is stored as a binary array
#
#
#                     #
#                     voxel_data = np.fromfile(file, dtype=np.uint8)
#                     original_shape = voxel_data.shape[0]
#                     zoom_factor = new_shape / original_shape
#                     voxel_data = zoom(voxel_data, zoom_factor)
#                     with open('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/test_voxel.obj', 'w')as f:
#                         voxel_data.export(file_obj=f, file_type='binvox')
#                 Person_GT[file_name] = voxel_data
#     return Person_GT


def read_teeth_mesh(mesh_path):
    Person_GT = {}
    for subdir, _, files in os.walk(mesh_path):
        files = sorted(files)
        for file in files:
            if file.endswith('.stl') and '._' in file:
                file_num = file.split('._')[0]
                file_name = int(file_num) * 5
                file_path = os.path.join(subdir, file)
                # 读取mesh
                mesh = trimesh.load(file_path)
                # 获取网格的点
                pts = mesh.vertices
                if pts.shape[0] < 4096:
                    pts, face_indices = trimesh.sample.sample_surface(mesh, count=4096)
                else:
                    # random choose 4096 points
                    indices = np.random.choice(pts.shape[0], size=4096, replace=False)
                    pts = pts[indices]
                pts = pts.astype('float32')
                # pts = pts.float()
                # 计算网格的法线
                # normals = mesh.vertex_normals
                # normals = normals.astype('float32')
                Person_GT[file_name] = pts
    return Person_GT


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        mask = mask.flatten()
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def evaluate_iou(pre, gt):
    pre = pre.squeeze(0)
    # 计算两个点云的最小包围盒
    min1, _ = torch.min(pre, dim=0)
    max1, _ = torch.max(pre, dim=0)
    min2, _ = torch.min(gt, dim=0)
    max2, _ = torch.max(gt, dim=0)

    # 计算两个最小包围盒的交集
    intersection_min = torch.max(min1, min2)
    intersection_max = torch.min(max1, max2)

    # 计算交集的体积
    intersection_volume = torch.prod(torch.clamp(intersection_max - intersection_min, min=0))

    # 计算两个最小包围盒的并集
    union_min = torch.min(min1, min2)
    union_max = torch.max(max1, max2)

    # 计算并集的体积
    union_volume = torch.prod(union_max - union_min)

    # 计算 IOU
    iou = intersection_volume / union_volume

    return iou

    # """
    #     计算两个点云之间的 IoU 值
    #     :param pred: 预测点云，形状为 (16, N, 3)
    #     :param gt: 真实点云，形状为 (1, N, 3)
    #     :return: IoU 值，形状为 (B,)
    #     """
    # # 将真实点云扩展到与预测输出相同的批量大小
    # gt = gt.expand(pred.shape[0], -1, -1)
    # # 计算两个点云之间的距离矩阵
    # dist_matrix = torch.cdist(pred, gt)
    # # 计算每个预测点与真实点之间的最小距离
    # min_dists, _ = torch.min(dist_matrix, dim=2)
    # # 定义阈值
    # threshold = 0.1
    # # 计算两个点云之间的匹配点数
    # matches = torch.sum(min_dists <= threshold, dim=1)
    # # 计算 IoU 值
    # iou = matches / (2 * pred.shape[1] - matches)
    # return iou


def fibonacci_sphere(samples=4096):
    points = np.zeros((samples, 3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points[i] = np.array([x, y, z])
        # to tensor
    points = torch.from_numpy(points)
    return points

if __name__ == '__main__':
    data_root = 'xxx/Project/3D_Dental_Master/PointNet_3d/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = ToothLoader(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()