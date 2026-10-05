import torch
import torch.nn.functional as F


def rec_loss(outputs, gt):
    loss = F.mse_loss(outputs, gt)
    return loss


def proj_loss(outputs, gt):
    loss_1 = F.mse_loss(torch.mean(outputs, dim=1), torch.mean(gt, dim=1))
    loss_2 = F.mse_loss(torch.mean(outputs, dim=2), torch.mean(gt, dim=2))
    loss_3 = F.mse_loss(torch.mean(outputs, dim=3), torch.mean(gt, dim=3))
    return (loss_1 + loss_2 + loss_3) / 3


def ce_loss(outputs, gt):
    loss = torch.nn.CrossEntropyLoss(outputs, gt)
    return loss

