####################
#
# Get Nrrd 的scale 都不一样 导致全景片大小很不一致
# 找出 spaciong 最多的那个数
# 归一化
# XXX
# 2023-10-23
# result = spacing值最多的数是0.3
#       spacing值最多的数出现了124次
####################

import os
import time
import trimesh
import open3d as o3d
import numpy as np
import copy
from collections import Counter
import nrrd
from setuptools import glob


##################################
#  读取一下原始cbct的nrrd 获得里面的space信息
##################################
def read_nrrd(nrrd_path):
    # load nrrd
    data, options = nrrd.read(nrrd_path)
    spacing = options['space directions']
    x = spacing[0][0]
    # y = spacing[1][1]
    # z = spacing[2][2]
    # translation = np.array([x, y, z])
    # print(translation)
    return data, spacing, options, x


if __name__ == '__main__':
    nrrd_path = "xxx/DATA/Seg_Teeth_nrrd/teeth/"
    spacing_list = []
    for root, dirs, files in os.walk(nrrd_path):
        for file in files:
            single_nrrd_path = nrrd_path + file
            data, spacing, options, x = read_nrrd(single_nrrd_path)
            spacing_list.append(x)

    most_common_spacing = Counter(spacing_list).most_common(1)[0][0]
    print(f"spacing值最多的数是{most_common_spacing}")
    most_common_count = Counter(spacing_list).most_common(1)[0][1]
    print(f"spacing值最多的数出现了{most_common_count}次")
