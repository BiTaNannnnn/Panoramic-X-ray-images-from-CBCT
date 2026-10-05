'''
XXX 2023.11.6
生成的全景片（L，D） L = 牙弓曲线长度，D = CBCT深度
由于每个cbct 的spacing 不一致，
Step 1 ： 统一 Spacing
Step 2 ： Resize CBCT 比例大小全景片
得到全景片真实大小
result = spacing值最多的数是0.3
        spacing值最多的数出现了124次
生成的全景片保存：
'''

import cv2
from utils.local_io import *
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


def proposed_img(file_path, spacing, most_common_spacing):
    # 加载图像
    img = cv2.imread(file_path)
    shape = img.shape
    print('img shape = ', shape)
    img_height, img_width = img.shape[:2]

    # 计算新的尺寸
    new_size_w, new_size_h = (int(img_width * spacing / most_common_spacing),
                              int(img_height * spacing / most_common_spacing))
    print('new size(w,h) = ', new_size_w, new_size_h)
    # img 用双三次差值
    img_scaled = cv2.resize(img, dsize=(new_size_w, new_size_h), interpolation=cv2.INTER_CUBIC)
    is_saved = cv2.imwrite(file_path, img_scaled)
    if is_saved:
        print('Image is successfully saved.')
    else:
        print('Image is not saved.')

def proposed_mask(file_path, spacing, most_common_spacing):
    # 加载图像
    img = cv2.imread(file_path)
    shape = img.shape
    print('img shape = ', shape)
    img_height, img_width = img.shape[:2]

    # 计算新的尺寸
    new_size_w, new_size_h = (int(img_width * spacing / most_common_spacing),
                              int(img_height * spacing / most_common_spacing))
    print('new size(w,h) = ', new_size_w, new_size_h)
    # mask 用邻近差值
    img_scaled = cv2.resize(img, dsize=(new_size_w, new_size_h), interpolation=cv2.INTER_NEAREST)
    voxel_grid_unique = np.unique(img_scaled)
    print(voxel_grid_unique)
    is_saved = cv2.imwrite(file_path, img_scaled)
    if is_saved:
        print('mask is successfully saved.')
    else:
        print('mask is not saved.')


if __name__ == '__main__':

    img_path = 'xxx/Project/3D_Dental_Master/DATA_SET/final_px_MeshScale/'  # 更改路径
    # img_save_path = 'xxx/Project/3D_Dental_Master/DATA_SET/final_px_MeshScale/'

    mask_path = 'xxx/Project/3D_Dental_Master/DATA_SET/final_label_px_MeshScale/'
    # mask_save_path = 'xxx/Project/3D_Dental_Master/DATA_SET/final_label_px_MeshScale/'
    files = os.listdir(img_path)
    total_n = len(files)
    for file in files:
        data_id = file.split('.', 2)[0]
        mask_name = data_id +'_mask.png'
        spacing = read_nrrd(data_id)
        most_common_spacing = 0.3
        if spacing == most_common_spacing:  # most_common_spacing
            continue
        else:
            # img resize
            # img_file = img_path + file
            # img = proposed_img(img_file, spacing, most_common_spacing)

            # mask resize
            mask_file = mask_path + mask_name
            mask = proposed_mask(mask_file, spacing, most_common_spacing)
    print('Finished All IDs !')
