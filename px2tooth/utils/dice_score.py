import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn.functional import cross_entropy, one_hot, softmax


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# add by myself
def Dice_loss(input: Tensor, target: Tensor, ep=1e-8):
    intersection = 2 * torch.sum(input * target) + ep
    union = torch.sum(input) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def combined_loss(input, target, weight_ce=0.7, weight_fl=0.3):
    ce_loss = nn.CrossEntropyLoss()
    fl_loss = FocalLoss()
    loss_ce = ce_loss(input, target)
    loss_fl = fl_loss(input, target)
    return weight_ce * loss_ce + weight_fl * loss_fl
###########################
#  组合loss 1 basnet_hybrid_loss
###########################


def ssim_loss(y_true, y_pred):
    """
       Structural Similarity Index (SSIM) loss
       """
    ssim_value = ssim(y_true, y_pred, window_size=11, size_average=True, full=False)
    return 1 - ssim_value


def IOU_loss(y_true, y_pred):
    """
    Intersection-Over-Union (IoU), also known as the Jaccard loss
    """
    return 1 - jacard_similarity(y_true, y_pred)


def jacard_similarity(y_true, y_pred):
    """
    Intersection-Over-Union (IoU), also known as the Jaccard Index
    """
    y_true_f = torch.flatten(y_true[0])
    y_pred_f = torch.flatten(y_pred[0])

    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection / union


def basnet_hybrid_loss(y_true, y_pred, n_classes):
    """
    Hybrid loss proposed in BASNET (https://arxiv.org/pdf/2101.04704.pdf)
    The hybrid loss is a combination of the binary cross entropy, structural similarity
    and intersection-over-union losses, which guide the network to learn
    three-level (i.e., pixel-, patch- and map- level) hierarchy representations.
    """
    bce_loss = nn.CrossEntropyLoss()
    bce_loss = bce_loss(y_true, y_pred)
    y_true = F.softmax(y_true, dim=1).float(),
    y_pred = F.one_hot(y_pred, n_classes).permute(0, 3, 1, 2).float(),

    ms_ssim_loss = ssim_loss(y_true, y_pred)
    jacard_loss = IOU_loss(y_true, y_pred)

    return bce_loss + ms_ssim_loss + jacard_loss



# ssim loss ha
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True, full=False):
    (_, channel, height, width) = img1[0].size()
    window = create_window(window_size, channel).to(img1[0].device)

    mu1 = F.conv2d(img1[0], window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2[0], window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1[0]*img1[0], window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2[0]*img2[0], window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1[0]*img2[0], window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

###########################
#  组合loss 2 FocalTverskyLoss
###########################

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=0.75, smooth=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # 将 y_true 转换为 one-hot 编码
        y_true = nn.functional.one_hot(y_true.long(), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        true_pos = torch.sum(y_true * y_pred)
        false_neg = torch.sum(y_true * (1 - y_pred))
        false_pos = torch.sum((1 - y_true) * y_pred)

        tversky = (true_pos + self.smooth) / (
                    true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        return focal_tversky


###########################
#  组合loss 3 BoundaryDoULoss
###########################


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).to('cuda:5')
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).to('cuda:5'),padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes


###########################
#  组合loss 4 TLoss
###########################
class TLoss(nn.Module):
    def __init__(
            self,
            config,
            device,
            nu: float = 1.0,
            epsilon: float = 1e-8,
            reduction: str = "mean",
    ):
        """
        Implementation of the TLoss.

        Args:
            config: Configuration object for the loss.
            nu (float): Value of nu.
            epsilon (float): Value of epsilon.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super().__init__()
        self.config = config
        self.D = torch.tensor(
            (200 * 340),
            dtype=torch.float,
            device=device,
        )

        self.lambdas = torch.ones(
            (200, 340),
            dtype=torch.float,
            device=device,
        )
        self.nu = nn.Parameter(
            torch.tensor(nu, dtype=torch.float, device=device)
        )
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=device)
        self.reduction = reduction

    def forward(
            self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Model's prediction, size (B x W x H).
            target_tensor (torch.Tensor): Ground truth, size (B x W x H).

        Returns:
            torch.Tensor: Total loss value.
        """

        input_tensor = torch.argmax(input_tensor, dim=1)
        delta_i = input_tensor - target_tensor
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = (
                first_term
                + second_term
                + third_term
                + fourth_term
                + fifth_term
                + sixth_term
        )

        if self.reduction == "mean":
            return total_losses.mean()
        elif self.reduction == "sum":
            return total_losses.sum()
        elif self.reduction == "none":
            return total_losses
        else:
            raise ValueError(
                f"The reduction method '{self.reduction}' is not implemented."
            )



###########################
#  组合loss 5 JDTLoss
###########################

class JDTLoss(_Loss):
    def __init__(self,
                 mIoUD=1.0,
                 mIoUI=0.0,
                 mIoUC=0.0,
                 alpha=0.5,
                 beta=0.5,
                 gamma=1.0,
                 smooth=1.0,
                 threshold=0.01,
                 active_classes_mode_hard="PRESENT",
                 active_classes_mode_soft="ALL",
                 class_weights=None,
                 ignore_index=None):
        """
        Arguments:
            mIoUD (float): The weight of the loss to optimize mIoUD.
            mIoUI (float): The weight of the loss to optimize mIoUI.
            mIoUC (float): The weight of the loss to optimize mIoUC.
            alpha (float): The coefficient of false positives in the Tversky loss.
            beta (float): The coefficient of false negatives in the Tversky loss.
            gamma (float): When `gamma` > 1, the loss focuses more on
                less accurate predictions that have been misclassified.
            smooth (float): A floating number to avoid `NaN` error.
            threshold (float): The threshold to select active classes.
            active_classes_mode_hard (str): The mode to compute
                active classes when training with hard labels.
            active_classes_mode_soft (str): The mode to compute
                active classes when training with hard labels.
            class_weights (list[float] | None): The weight of each class.
                If it is `list[float]`, its size should be equal to the number of classes.
            ignore_index (int | None): The class index to be ignored.

        Comments:
            Jaccard: `alpha`  = 1.0, `beta`  = 1.0
            Dice:    `alpha`  = 0.5, `beta`  = 0.5
            Tversky: `alpha` >= 0.0, `beta` >= 0.0
        """
        super().__init__()

        assert mIoUD >= 0 and mIoUI >= 0 and mIoUC >= 0 and \
               alpha >= 0 and beta >= 0 and gamma >= 1 and \
               smooth >= 0 and threshold >= 0
        assert active_classes_mode_hard in \
               ["ALL", "PRESENT", "PROB", "LABEL", "BOTH"]
        assert active_classes_mode_soft in \
               ["ALL", "PRESENT", "PROB", "LABEL", "BOTH"]
        assert class_weights == None or \
               all((isinstance(w, float)) for w in class_weights)
        assert ignore_index == None or \
               isinstance(ignore_index, int)

        self.mIoUD = mIoUD
        self.mIoUI = mIoUI
        self.mIoUC = mIoUC
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.threshold = threshold
        self.active_classes_mode_hard = active_classes_mode_hard
        self.active_classes_mode_soft = active_classes_mode_soft
        if class_weights == None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.tensor(class_weights)
        self.ignore_index = ignore_index


    def forward(self,
                logits,
                label,
                keep_mask=None):
        """
        Arguments:
            logits (torch.Tensor): Its shape should be (B, C, D1, D2, ...).
            label (torch.Tensor):
                If it is hard label, its shape should be (B, D1, D2, ...).
                If it is soft label, its shape should be (B, C, D1, D2, ...).
            keep_mask (torch.Tensor | None):
                If it is `torch.Tensor`,
                    its shape should be (B, D1, D2, ...) and
                    its dtype should be `torch.bool`.
        """
        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long

        logits = logits.view(batch_size, num_classes, -1)
        prob = logits.log_softmax(dim=1).exp()

        if keep_mask != None:
            assert keep_mask.dtype == torch.bool
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand(batch_size, num_classes, -1)
        elif self.ignore_index != None and hard_label:
            keep_mask = label != self.ignore_index
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand(batch_size, num_classes, -1)

        if hard_label:
            label = label.view(batch_size, -1)
            label = F.one_hot(
                torch.clamp(label, 0, num_classes - 1), num_classes=num_classes)
            label = label.permute(0, 2, 1).float()
            active_classes_mode = self.active_classes_mode_hard
        else:
            label = label.view(batch_size, num_classes, -1)
            active_classes_mode = self.active_classes_mode_soft

        assert prob.shape == label.shape and \
               (keep_mask == None or prob.shape == keep_mask.shape)

        loss = self.forward_loss(prob,
                                 label,
                                 keep_mask,
                                 active_classes_mode)

        return loss


    def forward_loss(self,
                     prob,
                     label,
                     keep_mask,
                     active_classes_mode):
        if keep_mask != None:
            prob = prob * keep_mask
            label = label * keep_mask

        cardinality = torch.sum(prob + label, dim=2)
        difference = torch.sum(torch.abs(prob - label), dim=2)
        intersection = (cardinality - difference) / 2
        fp = torch.sum(prob, dim=2) - intersection
        fn = torch.sum(label, dim=2) - intersection

        loss = 0
        batch_size, num_classes = prob.shape[:2]
        if self.mIoUD > 0:
            active_classes = self.compute_active_classes(prob,
                                                         label,
                                                         active_classes_mode,
                                                         num_classes,
                                                         (0, 2))
            loss_mIoUD = self.forward_loss_mIoUD(intersection,
                                                 fp,
                                                 fn,
                                                 active_classes)
            loss += self.mIoUD * loss_mIoUD

        if self.mIoUI > 0 or self.mIoUC > 0:
            active_classes = self.compute_active_classes(prob,
                                                         label,
                                                         active_classes_mode,
                                                         (batch_size, num_classes),
                                                         (2, ))
            loss_mIoUI, loss_mIoUC = self.forward_loss_mIoUIC(intersection,
                                                              fp,
                                                              fn,
                                                              active_classes)
            loss += self.mIoUI * loss_mIoUI + self.mIoUC * loss_mIoUC

        return loss


    def compute_active_classes(self,
                               prob,
                               label,
                               active_classes_mode,
                               shape,
                               dim):
        if active_classes_mode == "ALL":
            mask = torch.ones(shape, dtype=torch.bool)
        elif active_classes_mode == "PRESENT":
            mask = torch.amax(label, dim) > 0.5
        elif active_classes_mode == "PROB":
            mask = torch.amax(prob, dim) > self.threshold
        elif active_classes_mode == "LABEL":
            mask = torch.amax(label, dim) > self.threshold
        elif active_classes_mode == "BOTH":
            mask = torch.amax(prob + label, dim) > self.threshold

        active_classes = torch.zeros(shape,
                                     dtype=torch.bool,
                                     device=prob.device)
        active_classes[mask] = 1

        return active_classes


    def forward_loss_mIoUD(self,
                           intersection,
                           fp,
                           fn,
                           active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(intersection)

        intersection = torch.sum(intersection, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = (intersection + self.smooth) / \
            (intersection + self.alpha * fp + self.beta * fn + self.smooth)

        loss_mIoUD = 1.0 - tversky
        if self.gamma > 1:
            loss_mIoUD **= self.gamma
        if self.class_weights != None:
            loss_mIoUD *= self.class_weights

        loss_mIoUD = loss_mIoUD[active_classes]
        loss_mIoUD = torch.mean(loss_mIoUD)

        return loss_mIoUD


    def forward_loss_mIoUIC(self,
                            intersection,
                            fp,
                            fn,
                            active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(intersection), \
                   0. * torch.sum(intersection)

        tversky = (intersection + self.smooth) / \
            (intersection + self.alpha * fp + self.beta * fn + self.smooth)

        loss_matrix = 1.0 - tversky
        if self.gamma > 1:
            loss_matrix **= self.gamma
        if self.class_weights != None:
            class_weights = self.class_weights.unsqueeze(0).expand(loss_matrix.shape)
            loss_matrix *= class_weights

        loss_matrix *= active_classes
        loss_mIoUI = self.reduce(loss_matrix,
                                 active_classes,
                                 1)
        loss_mIoUC = self.reduce(loss_matrix,
                                 active_classes,
                                 0)

        return loss_mIoUI, loss_mIoUC


    def reduce(self,
               loss_matrix,
               active_classes,
               dim):
        loss = torch.sum(loss_matrix, dim)
        active_sum = torch.sum(active_classes, dim)
        active_dim = active_sum > 0
        loss = loss[active_dim] / active_sum[active_dim]
        loss = torch.mean(loss)

        return loss


###########################
#  组合loss 6 PolyLoss
###########################
class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):

        pt = outputs * targets

        return (self.epsilon * (1.0 - pt.sum(dim=1))).mean()



###########################
#  组合loss 7 focalLoss 多分类 + 根据label的面积做倒数加权
###########################


def FocalLoss_weight(logits, labels, device):
    """
    cal culates loss
    logits: batch_size * labels_length * seq_length
    labels: batch_size * seq_length
    """
    gamma = 2
    size_average = True
    elipson = 0.000001
    weight_ce = 1.0
    weight_fl = 5.0
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(logits, labels)
    if labels.dim() > 2:
        labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
        labels = labels.transpose(1, 2)
        labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
    if logits.dim() > 3:
        logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
        logits = logits.transpose(2, 3)
        logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
    assert (logits.size(0) == labels.size(0))
    assert (logits.size(2) == labels.size(1))
    batch_size = logits.size(0)
    labels_length = logits.size(1)
    seq_length = logits.size(2)

    # transpose labels into labels onehot
    new_label = labels.unsqueeze(1)
    # 指定cuda
    # device = torch.device("cuda:1")  # GPU 1
    label_onehot = torch.zeros([batch_size, labels_length, seq_length]).to(device).scatter_(1, new_label.to(device), 1)
    # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

    # calculate class weights
    class_weights = 1.0 / (label_onehot.sum(dim=[0, 2]) + elipson)
    # # 计算所有类别权重的总和
    total_weights = class_weights.sum()
    # # 对每个类别权重除以总和，确保所有类别权重的总和为1
    class_weights = class_weights / total_weights

    # calculate log
    log_p = F.log_softmax(logits, dim=1)
    # pt = torch.exp(log_p) # 原始版本 可试试
    pt = label_onehot * log_p # 改进版本
    sub_pt = 1 - pt
    fl = -class_weights.unsqueeze(0).unsqueeze(2) * (sub_pt) ** gamma * log_p
    if size_average:
        fl = fl.mean()
    else:
        fl = fl.sum()

    loss = weight_ce * loss_ce + weight_fl * fl
    return loss


###########################
#  组合loss 8 focalLoss +根据label的面积做倒数加权
###########################

def WeightedFocalLoss(inputs, targets):
    gamma = 2.
    # 计算每个类别的频率
    freqs = torch.bincount(targets.view(-1)).float() / targets.numel()
    freqs = torch.clamp(freqs, min=1e-6)

    # # 计算alpha参数
    alpha = freqs
    # alpha = 1.0 / freqs
    #
    # 计算每个像素的alpha值
    alpha = alpha[targets.view(-1)].view_as(targets)
    # 归一化
    # alpha = alpha / alpha.sum()

    # 计算多类别交叉熵损失
    CE_loss = F.cross_entropy(inputs, targets, reduction='none')

    # 计算Focal Loss
    pt = torch.exp(-CE_loss)
    F_loss = alpha * (1 - pt) ** gamma * CE_loss

    # 计算加权损失
    weighted_loss = (F_loss * alpha)
    loss = CE_loss + weighted_loss

    return loss.mean()

###########################
#  组合loss 9 focalLoss +固定alpha
###########################

def FocalLoss(inputs, targets, n_class, device):
    gamma = 2
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(inputs, targets)

    alpha = torch.ones(n_class) * 0.75
    alpha[0] = 0.25  # assuming class 0 is the background
    alpha = alpha.to(device)


    # 计算多类别交叉熵损失
    CE_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-CE_loss)
    F_loss = alpha[targets] * (1-pt)**gamma * CE_loss

    F_loss = torch.mean(F_loss)
    Loss = loss_ce + F_loss

    return Loss
