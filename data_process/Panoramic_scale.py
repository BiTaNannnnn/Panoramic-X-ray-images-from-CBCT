##############################
# generate px image 通过nrrd spacing得到真实大小的全景片
# XXX 2023.10.29
# Get 全景片 （真实大小）
# 得到全景片真实大小
###########################

import cv2
import open3d as o3d
from sympy import Point, Ray, Circle
import trimesh
import copy
from trimesh.voxel import creation
import matplotlib.pyplot as plt
from utils.local_io import *
from torchvision import transforms
from PIL import Image
from preprocess.arch_mask import *
from preprocess.skeleton import *
from preprocess.MPR import *
from tqdm import tqdm
from preprocess.projection import *


##################################
#  读取一下原始cbct的nrrd 获得里面的space信息
##################################
def read_nrrd(data_id):
    # load nrrd
    print('patient id = ', data_id)
    root_path = 'xxx/DATA/Seg_Teeth_nrrd/teeth'  # 分割之后的nrrd
    case_path = os.path.join(root_path, data_id + '_teeth_1022.seg.nrrd')
    print("\n case_path = " + case_path)
    options = nrrd.read(case_path)
    spacing = options[1]['space directions']
    x = spacing[0][0]
    y = spacing[1][1]
    z = spacing[2][2]
    spacing = x
    if x == y == z:
        print('xyz 相等', x)
    else:
        print('xyz不相等', x,y,z)
    # translation = np.array([x, y, z])
    # print(translation)
    return spacing

def proposed(file_path, spacing):
    scale = 1/spacing

    # 加载图像
    img = cv2.imread(file_path)
    shape = img.shape
    print(shape)
    img_height, img_width = img.shape[:2]

    # 计算新的尺寸
    new_size_w, new_size_h = (int(img_width / scale), int(img_height / scale))
    print(new_size_w, new_size_h)
    img_scaled = cv2.resize(img, dsize=(new_size_w, new_size_h), interpolation=cv2.INTER_NEAREST)
    file_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/scale.png'
    is_saved = cv2.imwrite(file_path, img_scaled)
    if is_saved:
        print('Image is successfully saved.')
    else:
        print('Image is not saved.')


if __name__ == '__main__':

    img_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train/imgs/'  # 更改路径
    files = os.listdir(img_path)
    total_n = len(files)
    for file in files:
        data_id = file.split('.', 2)[0]
        spacing = read_nrrd(data_id)
        img_file = img_path + file
        mask_file = str('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train/masks/' + data_id + '_mask.png')
        proposed(img_file, spacing)
        proposed(mask_file, spacing)

