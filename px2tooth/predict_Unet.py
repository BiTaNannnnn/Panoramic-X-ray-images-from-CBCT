'''
Unet 牙齿分割预测结果 + mIOU 计算

有两个方法
1//  训练的用all tooth +IOU 计算 all tooth
2//  训练的用all tooth +IOU 计算 32 颗牙齿 （55 以下的多生牙都不算）
3//  训练的用 ignore 多生牙 +IOU 计算 32 颗牙齿 （55 以下的多生牙都不算）
4// 训练的用 ignore 多生牙 +IOU 计算 28 颗牙齿

2 的iou 分数最高

'''


import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from utils.IOU import *




'''
cd xxx/Project/3D_Dental_Master/Pytorch-UNet-master

python predict_Unet.py -i xxx/Project/3D_Dental_Master/Pytorch-UNet-master/data/imgs/CASE_ID.png -o xxx/Project/3D_Dental_Master/Pytorch-UNet-master/data/test_out/CASE_ID_test111.png
'''
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # output = output[0]
        output = output.cpu()
        # print(output.shape)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    model_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/checkpoints/Seg_1211_512/'
    parser.add_argument('--model', '-m', default=model_path + 'checkpoint_epoch2080.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512/test/', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    parser.add_argument('--output', '-o', default='xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512/Result/Test_out_Seg1211_padding-2080lun/', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=48, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    # for i, v in mask_values.items():
    #     out[mask == i] = v

    # pre_result_values = np.unique(out)
    # print('pre_result_values = ', pre_result_values)
    # print(len(pre_result_values))

    return Image.fromarray(out)


def get_bbox(mask, padding=0):
    mask_gray = np.max(mask, axis=-1)
    # 找到mask中所有非零元素的坐标
    rows, cols = np.where(mask_gray != 0)
    # 计算边界框
    top = max(0, np.min(rows) - padding)
    bottom = min(mask_gray.shape[0], np.max(rows) + padding)
    left = max(0, np.min(cols) - padding)
    right = min(mask_gray.shape[1], np.max(cols) + padding)
    return top, bottom, left, right


def center_crop(img, mask, bbox, crop_size=(200, 340)):
    top, bottom, left, right = bbox
    bbox_width = right - left
    bbox_height = bottom - top
    # 计算裁剪区域的中心点
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    # 如果bbox的宽度大于crop_size的宽度，那么在bbox的宽度范围内随机选择一个起始点
    if bbox_width > crop_size[1]:
        start_x = np.random.randint(left, right - crop_size[1] + 1)
    else:
        # 如果bbox的宽度不大于crop_size的宽度，那么在bbox的中心周围随机选择一个起始点
        max_dx = min(center_x - left, right - center_x, crop_size[1] // 2 - bbox_width // 2)
        if max_dx > 0:
            dx = np.random.randint(-max_dx, max_dx)
        else:
            dx = 0
        start_x = max(0, left + dx)

    # 如果bbox的高度大于crop_size的高度，那么在bbox的高度范围内随机选择一个起始点
    if bbox_height > crop_size[0]:
        start_y = np.random.randint(top, bottom - crop_size[0] + 1)
    else:
        # 如果bbox的高度不大于crop_size的高度，那么在bbox的中心周围随机选择一个起始点
        max_dy = min(center_y - top, bottom - center_y, crop_size[0] // 2 - bbox_height // 2)
        if max_dy > 0:
            dy = np.random.randint(-max_dy, max_dy)
        else:
            dy = 0
        start_y = max(0, bottom + dy)
    # 计算裁剪区域的结束点
    end_x = start_x + crop_size[1]
    end_y = start_y + crop_size[0]
    # 检查是否超出图像边界
    if end_x > img.shape[1]:
        start_x -= end_x - img.shape[1]
        end_x = img.shape[1]
    if end_y > img.shape[0]:
        start_y -= end_y - img.shape[0]
        end_y = img.shape[0]
    # 裁剪图像和mask
    img_crop = img[start_y:end_y, start_x:end_x]
    mask_crop = mask[start_y:end_y, start_x:end_x]
    return img_crop, mask_crop



if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = args.output

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    # print(mask_values)
    # mask_values_dic = {i: v for i, v in enumerate(
    #     [0, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 55, 60, 65, 70, 75, 80, 85, 90, 105, 110, 115,
    #      120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225, 230, 235, 240])}
    mask_values_dic = {i: v for i, v in enumerate(
        [0, 55, 60, 65, 70, 75, 80, 85, 90, 105, 110, 115,
         120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225, 230, 235, 240])}
    # print(mask_values_dic)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    IOU_DIC = {}

    for root, dirs, files in os.walk(in_files):
        for file in files:
            ID = str(file.split('.')[0])
            # file = 'CASE_ID.png'
            filename = str(root) + str(file)
            # logging.info(f'Predicting image {filename} ...')
            img = Image.open(filename)
            # img = Image.open(in_files)

            dir_mask_file = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512/test_masks/'
            dir_mask = dir_mask_file + str(file.split('.')[0]) + '_mask.png'
            true_mask = Image.open(dir_mask)
            # test
            # img.save('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_all/' + ID + '.png')
            # true_mask.save('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_all/' + ID + '_mask.png')


            # ####################################
            # # 增加crop 操作
            # true_mask = np.asarray(true_mask)
            # img = np.asarray(img)
            # bbox = get_bbox(true_mask)
            # # img, mask = center_crop(img, true_mask, bbox)
            # img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            # true_mask = true_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]

            # img_save = Image.fromarray(img)
            # true_mask = Image.fromarray(true_mask)
            # img_save.save('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_all/test_crop/'+ ID +'.png')
            # true_mask.save('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_all/test_masks_crop/' + ID + '_mask.png')
            # ####################################

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            result_values = np.unique(mask)
            # print('pre_result_values = ', result_values)
            # print(len(result_values))
            # print('all mask values = ', mask_values)
            # print(len(mask_values))
            if not args.no_save:
                out_filename = out_files + str(file.split('.')[0]) + '_1211padding.png'
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                # logging.info(f'Mask saved to {out_filename}')


            # print('Seg 单独训练结果：')
            print('ID_name = ', str(file.split('.')[0]))
            # iou = img_iou(true_mask, result)
            # 计算mIoU
            miou = mIOU(result, true_mask)
            print('miou =', miou)
            IOU_DIC[ID] = miou

            true_mask_value = np.unique(true_mask)
            # print('true_mask_values =', true_mask_value)
            # print(len(true_mask_value))

            # # 设置要忽略的值 mask 已经映射成连续的values
            # ignore_values = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45]
            # miou = iou_ignore(result, true_mask)
            # print('miou =', miou)


            # 检查每个label + 位置是否对应
            true_mask = np.array(true_mask.convert('L'))
            result = np.array(result.convert('L'))

            result_value = np.unique(result)
            # print('result convert_values =', result_value)
            # print(len(result_value))
            # assert true_mask.shape == result.shape
            #
            # for label in mask_values_dic.values():
            #     label = 230
            #     # 创建一个新的mask，其中只包含当前标签的位置
            #     true_mask_label = np.where(true_mask == label, label, 0)
            #     pred_mask_label = np.where(result == label, label, 0)
            #
            #     # 显示结果
            #     fig, ax = plt.subplots(1, 2)
            #     ax[0].imshow(true_mask_label, cmap='gray')
            #     ax[0].set_title(f'True Mask for Label {label}')
            #     ax[1].imshow(pred_mask_label, cmap='gray')
            #     ax[1].set_title(f'Predicted Mask for Label {label}')
            #     plt.show()
            #
            # if args.viz:
            #     logging.info(f'Visualizing results for image {filename}, close to continue...')
            #     plot_img_and_mask(mask, true_mask)
    average_iou = sum(IOU_DIC.values()) / len(IOU_DIC)
    print(average_iou)