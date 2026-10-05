import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps

###############################################
# get bounding box and save
# get_bbox()

# 把图1变成新图片的第一个通道，想把图2变成新图片的第二个通道
# combine_channel()


# get single mask and save 嵌入get_bbox
# get_single_mask()

# 在全景片上截出相应bbox的牙齿 嵌入get_bbox
# get_px_bbox()
###############################################

def check_dir(dir_path, INFO=False):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        if INFO:
            print('Making new directory: %s' % dir_path)
    return dir_path

def join_path(path, *paths):
    return os.path.join(path, *paths)

def get_single_mask():
    # 读取二值图像
    main_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/'
    img = cv2.imread(main_path + 'data3d/masks/CASE_ID_mask.png', 0)


    voxel_grid = np.unique(img)
    print(voxel_grid)  # [  0  21  55  60  65  75
    # to list 变成列表形式有逗号。
    tooth_pixels = np.unique(voxel_grid[voxel_grid != 0]).tolist()
    print(tooth_pixels)  # [21, 55, 60, 65, 75,

    for tooth_pixel in tooth_pixels:
        # 找到某颗牙齿的mask
        img = cv2.imread(main_path + 'data3d/masks/CASE_ID_mask.png', 0)
        img[img != tooth_pixel] = 0
        voxel_grid = np.unique(img)
        print(voxel_grid)
        # show
        plt.imshow(img, cmap='gray')
        plt.show()
        # save
        num = str(tooth_pixel)
        path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/single_mask'
        single_mask_dir = check_dir(join_path(path, 'CASE_ID'))
        cv2.imwrite(single_mask_dir + '/' + num + '.png', img)
    print('finished -- ID ：CASE_ID', )


def get_bbox():
    # 读取二值图像
    main_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/'
    img = cv2.imread(main_path + 'data3d/masks/CASE_ID_mask.png', 0)

    # # 假设您的牙齿像素值为[10, 15, 20, 25]
    # tooth_pixels = [21,  55,  60,  65,  75,  80,  85,  90, 105, 110, 115, 125, 130, 135, 155, 160, 165, 175,
    #  180, 185, 190, 205, 210, 215, 225, 230, 235, 240]

    voxel_grid = np.unique(img)
    print(voxel_grid)  # [  0  21  55  60  65  75
    # to list 变成列表形式有逗号。
    tooth_pixels = np.unique(voxel_grid[voxel_grid != 0]).tolist()
    print(tooth_pixels)  # [21, 55, 60, 65, 75,


    for tooth_pixel in tooth_pixels:
        ################test
        tooth_pixel = 235
        #################

        # 找到所有等于tooth_pixel的像素点的坐标
        y_coords, x_coords = (img == tooth_pixel).nonzero()

        # 计算最小外接矩形
        rect = cv2.minAreaRect(np.column_stack((x_coords, y_coords)))

        # 绘制矩形
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 垂直的矩形框
        x_min = np.min(box[:, 0])
        y_min = np.min(box[:, 1])
        x_max = np.max(box[:, 0])
        y_max = np.max(box[:, 1])
        box_con = ((x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min))
        box_con = np.array(box_con)
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        # 将矩形内部的像素值设置为tooth_pixel
        img_box = cv2.drawContours(img, [box_con], 0, tooth_pixel, 2)
        # 对全景片进行裁剪bbox
        get_px_bbox(box_con)

    # 显示图像
    plt.imshow(img_box, cmap='gray')
    plt.show()
    save_path = main_path + 'Data_Tooth/label_box/CASE_ID_box.png'
    cv2.imwrite(save_path, img_box)
    print('finished -- ID ：CASE_ID', )


def get_px_bbox(box_con):
    # 读取两张图片
    # image_a = 全景片原图
    image_px = cv2.imread('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train/imgs/CASE_ID.png')
    image_mask = cv2.imread('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/single_mask/CASE_ID/235.png')
    # num = 第几颗牙
    num = str(235)
    # 裁剪 全景片 bbox
    x_min = np.min(box_con[:, 0])
    y_min = np.min(box_con[:, 1])
    x_max = np.max(box_con[:, 0])
    y_max = np.max(box_con[:, 1])
    image_px = image_px[y_min:y_max, x_min:x_max]
    image_mask = image_mask[y_min:y_max, x_min:x_max]
    # size = 512
    width = x_max - x_min
    height = y_max - y_min
    ratio = min(512 / width, 512 / height)
    new_size = (int(width * ratio), int(height * ratio))
    image_px = np.array(Image.fromarray(image_px).resize(new_size))
    image_mask = np.array(Image.fromarray(image_mask).resize(new_size, resample=Image.NEAREST))

    delta_w = 512 - new_size[1]
    delta_h = 512 - new_size[0]
    padding = ((delta_w // 2, delta_w - (delta_w // 2)), (delta_h // 2, delta_h - (delta_h // 2)), (0, 0))
    image_px = np.pad(image_px, padding, mode='constant')
    image_mask = np.pad(image_mask, padding, mode='constant')
    # 显示或保存结果
    plt.imshow(image_px, cmap='gray')
    plt.show()

    plt.imshow(image_mask, cmap='gray')
    plt.show()
    save_px_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/single_px_teeth'
    single_px_dir = check_dir(join_path(save_px_path, 'CASE_ID'))
    cv2.imwrite(single_px_dir + '/' + num + '.png', image_px)

    save_mask_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/single_mask'
    single_px_dir = check_dir(join_path(save_mask_path, 'CASE_ID'))
    cv2.imwrite(single_px_dir + '/' + num + '.png', image_mask)

def combine_channel():
    num = str(235)
    # image_a = mask
    image_a = cv2.imread('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/single_mask/CASE_ID/235.png')
    # image_b = 全景片
    image_b = cv2.imread('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/single_px_teeth/CASE_ID/235.png')
    # Get the height and width of the images
    height, width = image_a.shape[:2]

    # Create a new image with two channels
    image_c = np.zeros((height, width, 3), dtype=np.uint8)

    # Set the first channel to img a and the second channel to img b
    image_c[:, :, 0] = image_a[:, :, 0]
    voxel_grid = np.unique(image_c)  # test
    print(voxel_grid)

    image_c[:, :, 1] = image_b[:, :, 0]
    # plt.imshow(image_c, cmap='gray')
    # plt.show()
    save_mask_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/generate_single_teeth'
    single_dir = check_dir(join_path(save_mask_path, 'CASE_ID'))
    cv2.imwrite(single_dir + '/' + num + '.png', image_c)

    # return imgc


if __name__ == '__main__':
    # get bounding box and save
    # get_bbox()

    # 把图1变成新图片的第一个通道，想把图2变成新图片的第二个通道
    combine_channel()

    # get single mask and save 嵌入get_bbox
    # get_single_mask()

    # 在全景片上截出相应bbox的牙齿 嵌入get_bbox
    # get_px_bbox()





