import pickle
import trimesh
import numpy as np
from skimage import io, transform
import os, sys
import cv2
from PIL import Image
import open3d as o3d
import torchvision
import torch.utils.data as data
from torchvision.transforms import Resize
import logging
import numpy as np
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
        mesh_pos = [0., 0., -0.8]
        normalization = True
        resize_with_constant_border = False
        IMG_SIZE = 224
        # name = os.path.join(self.file_root)
        # print(name)

        # 读入.dat文件
        # name = 'xxx/Project/3D_Dental_Master/Pixel2Mesh-Pytorc-ZhaoTong/dataset/ShapeNetSmall/03636649/101d0e7dbd07d8247dfd6bf7196ba84d/rendering/00.dat'
        # data = pickle.load(open(name, "rb"), encoding='latin1')
        # pts1, normals1 = data[:, :3], data[:, 3:]

        # 读入mesh
        mesh = trimesh.load('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT/47._Root.stl')
        # 获取网格的点
        pts = mesh.vertices
        pts = pts.astype('float32')
        # 计算网格的法线
        normals = mesh.vertex_normals
        normals = normals.astype('float32')

        # img, pts, normals = data[0].astype('float32') / 255.0, data[1][:, :3], data[1][:, 3:]
        # img_path1 = 'xxx/Project/3D_Dental_Master/Pixel2Mesh-Pytorc-ZhaoTong/dataset/ShapeNetSmall/03636649/101d0e7dbd07d8247dfd6bf7196ba84d/rendering/00.png'
        img_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/generate_single_teeth/CASE_ID/235.png'
        # 读取图像
        # img1 = cv2.imread(img_path1)
        img = cv2.imread(img_path)
        img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
        # 交换通道数(h,w,c)-->(c,h,w)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.0

        pts -= np.array(mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        # img_normalized = self.normalize_img(img) if self.normalization else img

        return {

            "images": img,
            "points": pts,
            "normals": normals,
            "length": length
            # "labels": self.labels_map[label],
            # "filename": filename,
            # "images": img_normalized,
            # "images_orig": img,

        }

        # return img, pts, label



    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":

    # file_root = "xxx"
    GT_dir = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT'
    images_dir = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/generate_single_teeth'
    # dataloader = ShapeNet(file_root, 'train_list_small.txt')
    dataset = ToothData3d(GT_dir, images_dir)
    print("Load %d files!\n" % len(dataset))



    # print("Info for the first data3d:")
    # print("Image Shape: ", img.shape)
    # print("Point cloud shape: ", pts.shape)
    # # print("Normal shape: ", normals.shape)
    # # print("Class: ", idx_class[label])
    # print("File name: ", name)