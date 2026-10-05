import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from PIL import Image
import numpy as np
from sklearn import metrics, neighbors


def img_iou(img, pre):

    smooth = 1e-5
    pre = np.array(pre)
    img = np.array(img)
    # 计算前景的IoU
    output_ = pre >= 1
    target_ = img >= 1

    intersection_foreground = (output_ & target_).sum()
    union_foreground = (output_ | target_).sum()
    iou_foreground = (intersection_foreground + smooth) / (union_foreground + smooth)

    # 计算背景的IoU
    output_background = pre < 1
    target_background = img < 1
    intersection_background = (output_background & target_background).sum()
    union_background = (output_background | target_background).sum()
    iou_background = (intersection_background + smooth) / (union_background + smooth)

    # 计算mean IoU
    mean_iou = (iou_foreground + iou_background) / 2

    print('前景IOU:{:.3f}%\n背景IOU:{:.3f}%\nMeanIou:{:.3f}%'.format(iou_foreground * 100, iou_background * 100,
                                                                 mean_iou * 100))

    return iou_foreground, iou_background, mean_iou

    # ####
    #
    # intersection = (output_ & target_).sum()
    # union = (output_ | target_).sum()
    # iou = (intersection + smooth) / (union + smooth)
    #
    # print('前景IOU:{:.3f}%\n'.format(iou * 100))
    #
    # return iou

def iou_ignore(input, target):
    n_classes = 48
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for label in range(n_classes):
        # if label in ignore_labels:
        #     continue  # 跳过你想忽略的标签
        pred_mask = (input == label)
        target_mask = (target == label)
        intersection[label] = np.logical_and(pred_mask, target_mask).sum()
        union[label] = np.logical_or(pred_mask, target_mask).sum()

    ious = intersection / (union + 1e-10)  # 避免除以零
    miou = np.mean(ious)  # 计算平均 IOU（mIOU）
    return miou

def mIOU(pred, target):
    ious = []
    # ignore value: 32 颗牙齿
    # ignore_value = [0, 55, 60, 65, 70, 75, 80, 85, 90, 105, 110, 115,
    #  120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225, 230, 235, 240]

    # ignore value: 28 颗牙齿
    ignore28_value = [0, 55, 60, 65, 70, 75, 80, 85,  105, 110, 115,
                    120, 125, 130, 135,  155, 160, 165, 170, 175, 180, 185,    205, 210, 215, 220, 225, 230, 235
                   ]

    # all value
    # all_value = [0, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 55, 60, 65, 70, 75, 80, 85, 90, 105, 110, 115,
    #                 120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225, 230, 235, 240]
    target = np.array(target.convert('L'))
    pred = np.array(pred.convert('L'))

    # if ignore value: true mask will - ignore value
    # 创建一个新的mask，其中只包含ignore_value中的值
    target = np.where(np.isin(target, ignore28_value), target, 0)
    pre_result_values = np.unique(target)

    for cls in ignore28_value:
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))  # 如果分母为0，则IOU为nan
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


        #################################
    # im = np.array(pre)
    # im_gt = np.array(img)
    #
    # im[im > 0] = 1
    # im_gt[im_gt > 0] = 1
    #
    # union = im * im_gt
    # iou = union.sum() / (im.sum() + im_gt.sum() - union.sum())
    #
    # print('前景IOU:{:.3f}%\n'.format(iou * 100))
    ##############################################

    # im = np.array(pre)
    # im_gt = np.array(img)
    #
    # c = metrics.confusion_matrix(im.flatten(), im_gt.flatten())  # 混淆矩阵
    # TP = c[1][1]  # 预测为前景，GT为前景
    # TN = c[0][0]  # 预测为前景，GT为前景
    # FP = c[1][0]  # 预测为前景，GT为背景
    # FN = c[0][1]  # 预测为背景，GT为前景
    # # print('IOU统计:\n'+str(c)+'\n\nFP为:'+str(FP)+'  FN为:'+str(FN)+'  TP为:'+str(TP)+'  TN:'+str(TN))
    #
    # iou_p = TP/(TP+FN+FP)  # 前景的IOU
    # iou_n = TN/(TN+FN+FP)  # 背景的IOU
    # mean_iou = (iou_n+iou_p)/2
    #
    # print('前景IOU:{:.3f}%\n背景IOU:{:.3f}%\nMeanIou:{:.3f}%'.format(iou_p*100, iou_n*100, mean_iou*100))

    # return iou_p, iou_n, mean_iou


if __name__ == '__main__':
    str1 = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/data/test_out/CASE_ID_test.png'
    input = Image.open(str1)
    str2 = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/data/masks/CASE_ID_mask.png'
    target = Image.open(str2)

    # 设置要忽略的值 mask 已经映射成连续的values
    ignore_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    iou_ignore(input, target, ignore_values)
    # img_iou(im_gt, im)