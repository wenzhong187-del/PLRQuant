# -*- coding: utf-8 -*-
"""
Utility Functions for PLRQuant
Includes Losses, Metrics, and Visualization tools.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import jaccard_score


# =======================================================================
#      Loss Functions (Focused on Binary Segmentation)
# =======================================================================

class WeightedBCE(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        loss = torch.nn.functional.binary_cross_entropy(logit, truth, reduction='none')

        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12

        loss = (self.weights[0] * pos * loss / pos_weight +
                self.weights[1] * neg * loss / neg_weight).sum()
        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        p = logit.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        w = t.detach()
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]

        p = w * p
        t = w * t
        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice = 1 - (2 * intersection + smooth) / (union + smooth)
        return dice.mean()


class BinaryDiceBCE(nn.Module):
    """
    针对瞳孔分割优化的联合损失函数：Dice + BCE
    """

    def __init__(self, dice_weight=1, BCE_weight=1):
        super(BinaryDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        """计算硬 Dice 系数用于进度打印"""
        with torch.no_grad():
            inputs = (inputs > 0.5).float()
            targets = (targets > 0.5).float()
            intersection = (inputs * targets).sum()
            dice = (2. * intersection + 1e-5) / (inputs.sum() + targets.sum() + 1e-5)
        return dice

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        return self.dice_weight * dice + self.BCE_weight * BCE


# =======================================================================
#      Evaluation Metrics
# =======================================================================

def dice_coef(y_true, y_pred):
    """计算单张图的 Dice 系数"""
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


# =======================================================================
#      Visualization Tools
# =======================================================================

def save_on_batch(images, masks, preds, names, vis_path):
    """
    保存预测结果对比图
    images: 原始图像 (B, C, H, W)
    masks: 真实标签 (B, 1, H, W)
    preds: 预测结果 (B, 1, H, W)
    """
    for i in range(preds.shape[0]):
        # 转换预测图和标签为 0-255 的 uint8
        pred_tmp = (preds[i][0].cpu().detach().numpy() > 0.5).astype(np.uint8) * 255
        mask_tmp = (masks[i][0].cpu().detach().numpy() > 0.5).astype(np.uint8) * 255

        # 处理原图
        img_tmp = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        img_tmp = (img_tmp * 255).astype(np.uint8)
        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)

        # 保存结果
        base_name = names[i].split('.')[0]
        cv2.imwrite(os.path.join(vis_path, f"{base_name}_orig.jpg"), img_tmp)
        cv2.imwrite(os.path.join(vis_path, f"{base_name}_pred.png"), pred_tmp)
        cv2.imwrite(os.path.join(vis_path, f"{base_name}_gt.png"), mask_tmp)
