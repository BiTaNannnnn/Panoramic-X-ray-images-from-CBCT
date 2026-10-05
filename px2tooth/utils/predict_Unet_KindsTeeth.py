'''

每类牙齿的分割的mIOU 箱状图
计算test dataset 中每类牙号 1-7 + 8 号牙齿的 平均miou

'''


from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np


def load_images_from_folder(folder, suffix):
    images = []
    # 按序读取 列表进行排序
    filenames = sorted([filename for filename in os.listdir(folder) if filename.endswith(suffix)])
    for filename in filenames:
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img = np.array(img.convert('L'))
            images.append(img)
    return images

def calculate_miou_per_label(preds, targets, labels):
    miou_dict = {label: [] for label in labels}
    for pred, target in zip(preds, targets):
        for label in labels:
            pred_mask = pred == label
            target_mask = target == label
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            if union == 0:
                miou = None
                print('union = 0')
            else:
                miou = intersection / union
                if miou == 0:
                    print(label, intersection, pred_mask.shape)
            miou_dict[label].append(miou)
    return miou_dict



if __name__ == '__main__':
    preds_folder = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_all/Result/Test_out_Seg1211_padding/'
    targets_folder = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_all/test_masks_padding/'

    preds = load_images_from_folder(preds_folder, '_1211padding.png')
    targets = load_images_from_folder(targets_folder, '_mask.png')

    labels = [0, 55, 60, 65, 70, 75, 80, 85, 90, 105, 110, 115, 120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225, 230, 235, 240]
    # preds, targets 是你的预测和目标mask，它们应该是PyTorch张量或者NumPy数组的列表，长度为25
    # 你需要在所有测试数据集上运行以下代码：
    miou_per_label = calculate_miou_per_label(preds, targets, labels)
    # Calculate the sum of miou scores for each label
    miou_sum_per_label = {label: np.mean([score for score in scores if score is not None]) for label,
                                                                        scores in miou_per_label.items()}
    print(miou_sum_per_label)

    # 计算28颗牙齿的平均结果
    # ignore_labels = [0, 90, 140, 190, 240]
    # 计算0 + 28 颗牙齿结果
    ignore_labels = [90, 140, 190, 240]
    miou_sum_selected_labels = {label: score for label, score in miou_sum_per_label.items() if label not in ignore_labels}
    average = np.mean(list(miou_sum_selected_labels.values()))
    print("average = ", average)


    # 将字典转换为列表，以便于绘图
    # all tooth 32 颗牙齿+背景
    labels_list = list(miou_per_label.keys())
    miou_list = list(miou_per_label.values())

    # 28 颗牙齿
    label_to_remove = [0, 90, 140, 190, 240]
    labels_list_new = [label for label in labels_list if label not in label_to_remove]
    miou_list_new = [miou for label, miou in zip(labels_list, miou_list) if label not in label_to_remove]

    plt.figure(figsize=(12, 6))
    plt.boxplot(miou_list_new, labels=labels_list_new)
    plt.title('mIoU for each tooth label')
    plt.xlabel('Tooth label')
    plt.ylabel('mIoU')
    plt.show()