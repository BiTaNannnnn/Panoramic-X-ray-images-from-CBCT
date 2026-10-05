from PIL import Image
import numpy as np
import trimesh
import torch
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from torch.utils.data import Dataset
from skimage import io, transform
from os.path import splitext, isfile, join
import cv2
from os import listdir
from torchvision.transforms import ToTensor
from pathlib import Path
import os

class Single_TeethLoader(Dataset):
    def __init__(self, split='train', data_root='xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT/',
                num_point=4096, img_scale=1.0, test_area=5, block_size=1.0, sample_rate=1.0, transform=None, mask_suffix: str = '_mask'):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        assert 0 < img_scale <= 1, 'Scale must be between 0 and 1'
        self.scale = img_scale
        self.mask_suffix = mask_suffix
        dir_img = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_test/imgs/' # Data_Tooth
        dir_mask = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_test/masks/'
        self.images_dir = Path(dir_img)
        self.mask_dir = Path(dir_mask)
        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if isfile(join(self.images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        # gt_root = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT'
        # self.ids = [splitext(file)[0] for file in listdir(gt_root) if
        #         isfile(join(gt_root, file)) and not file.startswith('.')]
        print(f'Creating dataset with {len(self.ids)} examples')
        print('Scanning mask files to determine unique values')
        unique = []
        for idx in self.ids:
            result = unique_mask_values(idx, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix)
            unique.append(result)
        # mask_values process
        unique_values = np.unique(np.concatenate(unique), axis=0)
        non_zero_values = unique_values[unique_values != 0]
        self.mask_values = list(sorted(non_zero_values.tolist()))
        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        print(f'Unique mask values: {self.mask_values}')

    def __getitem__(self, idx):
        # room_idx = self.room_idxs[idx]
        # points = self.room_points[room_idx]   # N * 6
        # labels = self.room_labels[room_idx]   # N
        # N_points = points.shape[0]

        IMG_SIZE = 512

        # 高斯噪声生成points
        points = torch.rand(16, 4096, 3)

        # 读入mesh
        mesh = trimesh.load('xxx/DATA/CBCT_data_2021_8_13_Processed/CASE_ID/11._Root.stl')
        # 获取网格的点
        pts = mesh.vertices
        # random choose 4096 points
        indices = np.random.choice(pts.shape[0], size=4096, replace=False)
        pts = pts[indices]
        pts = pts.astype('float32')
        # 计算网格的法线
        normals = mesh.vertex_normals
        normals = normals.astype('float32')

        GT_points = pts
        # # 拼接
        # current_points = np.zeros((pts.shape[0], 6), dtype='float32')
        # current_points[:, :3] = pts
        # current_points[:, 3:] = normals

        # labels
        current_labels = 47

        img_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/generate_single_teeth/CASE_ID/235.png'
        # 读取图像
        # img1 = cv2.imread(img_path1)
        img = cv2.imread(img_path)
        img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
        # 交换通道数(h,w,c)-->(c,h,w)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.0

        # assert pts.shape[0] == normals.shape[0]
        # length = pts.shape[0]

        return points, GT_points



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


def evaluate_iou(pred, gt):

    """
        计算两个点云之间的 IoU 值
        :param pred: 预测点云，形状为 (16, N, 3)
        :param gt: 真实点云，形状为 (1, N, 3)
        :return: IoU 值，形状为 (B,)
        """

    # 将真实点云扩展到与预测输出相同的批量大小
    gt = gt.expand(pred.shape[0], -1, -1)

    # 计算两个点云之间的距离矩阵
    dist_matrix = torch.cdist(pred, gt)

    # 计算每个预测点与真实点之间的最小距离
    min_dists, _ = torch.min(dist_matrix, dim=2)

    # 定义阈值
    threshold = 0.1

    # 计算两个点云之间的匹配点数
    matches = torch.sum(min_dists <= threshold, dim=1)

    # 计算 IoU 值
    iou = matches / (2 * pred.shape[1] - matches)

    return iou

if __name__ == '__main__':
    data_root = 'xxx/Project/3D_Dental_Master/PointNet_3d/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = Single_TeethLoader(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
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