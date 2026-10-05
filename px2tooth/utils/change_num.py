##################
# 把mask里= 13，14，23，24，33，45，43 的值= 0
##################
import logging
import numpy as np
import os
from PIL import Image
import cv2
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    else:
        return Image.open(filename)

if __name__ == '__main__':

    myparth = 'xxx/Project/3D_Dental_Master/DATA_SET/final_label_px/'  # 更改路径
    # myparth = 'xxx/DATA/nii_gz/nii_raw/'
    files = os.listdir(myparth)
    total_n = len(files)
    for file in files:
        data_id = file
        patient_path = str(myparth + data_id)
        img = cv2.imread(patient_path)
        mask = np.asarray(load_image(patient_path))
        unique = np.unique(mask)

        if 13 in unique:
            print(unique)
            print(data_id)
            img[img == 13] = 0
            cv2.imwrite(patient_path, img)

        elif 14 in unique:
            print(unique)
            print(data_id)
            img[img == 14] = 0
        elif 23 in unique:
            print(unique)
            print(data_id)
            img[img == 23] = 0
        elif 24 in unique:
            print(unique)
            print(data_id)
            img[img == 24] = 0
        elif 33 in unique:
            print(unique)
            print(data_id)
            img[img == 33] = 0
        elif 34 in unique:
            print(unique)
            print(data_id)
            img[img == 34] = 0
        elif 43 in unique:
            print(unique)
            print(data_id)
            img[img == 43] = 0
        elif 44 in unique:
            print(unique)
            print(data_id)
            img[img == 44] = 0
        elif 45 in unique:
            print(unique)
            print(data_id)
            img[img == 45] = 0

        else:
            continue
    print("finished all！")