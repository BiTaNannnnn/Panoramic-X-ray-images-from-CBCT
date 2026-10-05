import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.dice_score import dice_loss,    FocalLoss

def iou_score(output, target):
    smooth = 1e-6
    if torch.is_tensor(output):
        output = torch.sigmoid(output)
    intersection = (output * target).sum()
    total = (output + target).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return IoU


def mIOU_old(pred, target):
    ious = []
    smooth = 1e-6
    # # ignore value: 32 颗牙齿
    # ignore_value = [0, 55, 60, 65, 70, 75, 80, 85, 90, 105, 110, 115,
    #  120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225, 230, 235, 240]
    #
    # # ignore value: 28 颗牙齿
    # ignore28_value = [0, 55, 60, 65, 70, 75, 80, 85,  105, 110, 115,
    #                 120, 125, 130, 135,  155, 160, 165, 170, 175, 180, 185,    205, 210, 215, 220, 225, 230, 235
    #                ]
    #
    # # all value
    # all_value = [0, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 55, 60, 65, 70, 75, 80, 85, 90, 105, 110, 115,
    #                 120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225, 230, 235, 240]
    # target = np.array(target.convert('L'))
    # pred = np.array(pred.convert('L'))
    #
    # # if ignore value: true mask will - ignore value
    # # 创建一个新的mask，其中只包含ignore_value中的值
    # target = np.where(np.isin(target, all_value), target, 0)
    pre_result_values = np.unique(target)

    for cls in pre_result_values:
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        ious.append((intersection + smooth) / (union + smooth))
    return np.nanmean(ious)


def mIOU(pred, target, n_classes):
    ious = []
    pred = pred.argmax(dim=1)  # get the prediction results
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = torch.logical_and(pred_inds, target_inds).sum().float().item()
        union = torch.logical_or(pred_inds, target_inds).sum().float().item()
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)  # return mean IoU score over all classes



@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    miou_total = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true_copy = mask_true

            # predict the mask
            mask_pred = net(image)
            mask_pred_copy = mask_pred

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # # 计算loss= mask_true_copy
                # evl_loss = criterion(mask_pred_copy, mask_true_copy)
                # evl_loss += dice_loss(
                #     F.softmax(mask_pred_copy, dim=1).float(),
                #     F.one_hot(mask_true_copy, net.n_classes).permute(0, 3, 1, 2).float(),
                #     multiclass=True
                # )


                # 组合loss
                # evl_loss = basnet_hybrid_loss(mask_pred_copy, mask_true_copy, net.n_classes)

                # boundary_loss = BoundaryDoULoss(net.n_classes)
                # # evl_loss = boundary_loss(mask_pred_copy, mask_true_copy)
                # evl_loss = criterion(mask_pred_copy, mask_true_copy)
                # evl_loss += boundary_loss(mask_pred_copy, mask_true_copy)
                # Tloss
                # Tloss_loss = TLoss(net.parameters(), device, nu=1.0, epsilon=1e-8)

                # JDTLoss_loss = JDTLoss(alpha=0.5, beta=0.5)
                # evl_loss = criterion(mask_pred_copy, mask_true_copy)
                # evl_loss += JDTLoss_loss(mask_pred_copy, mask_true_copy)

                # # PolyLoss_loss = PolyLoss()
                # evl_loss = criterion(mask_pred_copy, mask_true_copy)
                # evl_loss += dice_loss(
                #     F.softmax(mask_pred_copy, dim=1).float(),
                #     F.one_hot(mask_true_copy, net.n_classes).permute(0, 3, 1, 2).float(),
                #     multiclass=True
                # )
                # # evl_loss += PolyLoss_loss(F.softmax(mask_pred_copy, dim=1).float(),
                # #                       F.one_hot(mask_true_copy, net.n_classes).permute(0, 3, 1, 2).float())

                # focal loss
                # evl_loss = criterion(mask_pred_copy, mask_true_copy)
                evl_loss = dice_loss(
                    F.softmax(mask_pred_copy, dim=1).float(),
                    F.one_hot(mask_true_copy, net.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                evl_loss += FocalLoss(mask_pred_copy, mask_true_copy, net.n_classes, device)

                miou = mIOU(mask_pred_copy, mask_true_copy, net.n_classes)
                miou_total += miou





    net.train()
    dice_score = dice_score / max(num_val_batches, 1)

    return miou_total / max(num_val_batches, 1), dice_score, evl_loss