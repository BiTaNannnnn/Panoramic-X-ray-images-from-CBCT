import numpy as np
import torch
import os, sys
import cv2
from PIL import Image
import open3d as o3d
import torchvision
import torch.utils.data as data
from torchvision.transforms import Resize
import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm



class ToothData3d(data.Dataset):
    """Dataset wrapping images and target meshes for ShapeNet dataset.

    Arguments:
    """


    def __init__(self, images_dir: str, GT_dir: str, scale: float = 1.0,
                 mask_suffix: str = '.stl'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(GT_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')



    def __getitem__(self, idx):
        img_name =  '235.png'     #  self.ids[idx]
        GT_name = self.ids[idx]
        # GT_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # img_file = list(self.images_dir.glob(name + '.*'))
        GT_file = 'xxx/DATA/CBCT_data_2021_8_13_Processed/CASE_ID/47._Root.stl'
        img_file= 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/generate_single_teeth/CASE_ID/235.png'
        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {img_name}: {img_file}'
        # assert len(GT_file) == 1, f'Either no GT or multiple GT found for the ID {GT_name}: {GT_file}'
        # 读取mesh
        mesh = o3d.io.read_triangle_mesh(GT_file)
        # 顶点
        vertices = np.asarray(mesh.vertices)
        pts_gt = torch.from_numpy(vertices)
        img = cv2.imread(img_file)
        img = np.asarray(img)
        # 交换通道数(h,w,c)-->(c,h,w)
        img = np.transpose(img, (2, 0, 1))
        # assert img.size == mask.size, \
        #     f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        name = '11._Root.stl'
        label = name.split('.')[0]
        # print(img.shape)
        # print(mask.shape)
        # print(np.unique(mask))
        # exit()
        return {
            'image': torch.as_tensor(img).float().contiguous(),
            'pts_gt': torch.as_tensor(pts_gt).long().contiguous(),
            'label': label
        }

        # name = os.path.join(self.file_root, self.file_names[index])
        # print(name)
        # # name = 'xxx/ShapeNet/03001627_1a6f615e8b1b5ae4dbbc9440457e303e_00.dat'
        # # data = pickle.load(open(name, "rb"), encoding = 'latin1')
        # # img, pts, normals = data[0].astype('float32') / 255.0, data[1][:, :3], data[1][:, 3:]
        # # pts, normals = data[1][:, :3], data[1][:, 3:]
        # img_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/generate_single_teeth/CASE_ID/235.png'
        # # 读取图像
        # img = cv2.imread(img_path)
        # img = np.asarray(img)
        # # 交换通道数(h,w,c)-->(c,h,w)
        # img = np.transpose(img, (2, 0, 1))
        #
        # # 读取三维网格文件
        #
        # # label = word_idx[self.file_names[index].split('_')[0]]
        # name = '11._Root.stl'
        # label = name.split('.')[0]
        # return img, pts, label, self.file_names[index]


    def __len__(self):
        return len(self.ids)



if __name__ == "__main__":

    # file_root = "xxx"
    GT_dir = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT'
    images_dir = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/generate_single_teeth/CASE_ID'
    # dataloader = ShapeNet(file_root, 'train_list_small.txt')
    dataset = ToothData3d(GT_dir, images_dir)
    print("Load %d files!\n" % len(dataset))



    # print("Info for the first data:")
    # print("Image Shape: ", img.shape)
    # print("Point cloud shape: ", pts.shape)
    # # print("Normal shape: ", normals.shape)
    # # print("Class: ", idx_class[label])
    # print("File name: ", name)