# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import weakref
from sklearn.metrics import roc_auc_score, jaccard_score
from torch.optim.optimizer import Optimizer
from functools import wraps

# =============================================================
# 1. 拓扑损失核心组件 (clDice 支撑函数 - 针对 OCTA 血管连通性)
# =============================================================

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    return img

def soft_skeletonize(x, iters=3):
    for _ in range(iters):
        sure_fg = soft_erode(x)
        x = torch.max(x - soft_erode(sure_fg), torch.zeros_like(x))
    return x

class SoftclDice(nn.Module):
    def __init__(self, iter=3, smooth=1.):
        super(SoftclDice, self).__init__()
        self.iter = iter
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 3:
            y_true = y_true.unsqueeze(1)
        if len(y_pred.shape) == 3:
            y_pred = y_pred.unsqueeze(1)
        skel_pred = soft_skeletonize(y_pred, iters=self.iter)
        skel_true = soft_skeletonize(y_true, iters=self.iter)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        return 1. - 2.0 * (tprec * tsens) / (tprec + tsens)

# =============================================================
# 2. 基础损失函数 (针对二值分割：瞳孔、OCTA)
# =============================================================

class WeightedBCE(nn.Module):
    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        return (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = logit.size(0)
        p = logit.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        w = truth.detach().view(batch_size, -1)
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
        p, t = w * p, w * t
        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        return (1 - (2 * intersection + smooth) / (union + smooth)).mean()

class WeightedDiceBCE_clDice(nn.Module):
    def __init__(self, dice_weight=0.4, BCE_weight=0.4, clDice_weight=0.2):
        super().__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.clDice_loss = SoftclDice(iter=3)
        self.weights = [dice_weight, BCE_weight, clDice_weight]

    def forward(self, inputs, targets):
        return self.weights[0] * self.dice_loss(inputs, targets) + \
               self.weights[1] * self.BCE_loss(inputs, targets) + \
               self.weights[2] * self.clDice_loss(inputs, targets)

    def _show_dice(self, inputs, targets):
        tmp_inputs = (inputs.detach() > 0.5).float()
        return 1.0 - self.dice_loss(tmp_inputs, targets)

# =============================================================
# 3. 多类损失函数 (针对 OCT 7类层级分割)
# =============================================================

class MultiClassDiceLoss(nn.Module):
    def __init__(self, n_classes=7, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logit, target):
        target = target.squeeze(1) if len(target.shape) == 4 else target
        target_one_hot = F.one_hot(target.long(), num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logit, dim=1)
        intersection = torch.sum(probs * target_one_hot, dim=(0, 2, 3))
        union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class MultiTaskOcuLoss(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(n_classes=n_classes)

    def forward(self, inputs, targets):
        targets = targets.squeeze(1).long() if len(targets.shape) == 4 else targets.long()
        return 0.5 * self.ce(inputs, targets) + 0.5 * self.dice(inputs, targets)

    def _show_dice(self, inputs, targets):
        return 1.0 - self.dice(inputs, targets)

# =============================================================
# 4. 评价指标自适应修正 (解决 Dice > 1 问题)
# =============================================================

def dice_coef_binary(y_true, y_pred):
    smooth = 1e-5
    y_true_f, y_pred_f = y_true.flatten(), y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multiclass(y_true, y_pred, n_classes):
    """ 正确计算多类分割的平均 Dice (Macro-Dice) """
    dice = np.zeros(n_classes)
    for i in range(n_classes):
        gt = (y_true == i).astype(np.float32)
        pd = (y_pred == i).astype(np.float32)
        intersection = np.sum(gt * pd)
        union = np.sum(gt) + np.sum(pd)
        dice[i] = (2. * intersection + 1e-5) / (union + 1e-5) if union > 0 else 1.0
    print(f"各层 Dice: {['%.4f' % d for d in dice]}")
    return np.mean(dice)

def dice_on_batch(masks, pred):
    dices = []
    n_classes = pred.shape[1]
    for i in range(pred.shape[0]):
        gt_np = masks[i].cpu().detach().numpy()
        if n_classes == 1:
            pd_np = (pred[i][0].cpu().detach().numpy() >= 0.5).astype(np.float32)
            dices.append(dice_coef_binary(gt_np, pd_np))
        else:
            pd_np = torch.argmax(pred[i], dim=0).cpu().detach().numpy()
            dices.append(dice_coef_multiclass(gt_np, pd_np, n_classes))
    return np.mean(dices)

def iou_on_batch(masks, pred):
    ious = []
    n_classes = pred.shape[1]
    for i in range(pred.shape[0]):
        gt_np = masks[i].cpu().detach().numpy().flatten()
        if n_classes == 1:
            pd_np = (pred[i][0].cpu().detach().numpy() >= 0.5).astype(np.uint8).flatten()
        else:
            pd_np = torch.argmax(pred[i], dim=0).cpu().detach().numpy().flatten()
        # 使用 macro 平均，不被背景面积主导
        ious.append(jaccard_score(gt_np, pd_np, average='macro', labels=list(range(n_classes))))
    return np.mean(ious)

def save_on_batch(images1, masks, pred, names, vis_path):
    n_classes = pred.shape[1]
    for i in range(pred.shape[0]):
        mask_np = masks[i].cpu().detach().numpy()
        if n_classes == 1:
            pred_img = (pred[i][0].cpu().detach().numpy() >= 0.5).astype(np.uint8) * 255
            gt_img = (mask_np > 0).astype(np.uint8) * 255
        else:
            res_np = torch.argmax(pred[i], dim=0).cpu().detach().numpy()
            scale = 255 // (n_classes - 1)
            pred_img = (res_np * scale).astype(np.uint8)
            gt_img = (mask_np * scale).astype(np.uint8)
        cv2.imwrite(f"{vis_path}{names[i][:-4]}_pred.png", pred_img)
        cv2.imwrite(f"{vis_path}{names[i][:-4]}_gt.png", gt_img)

def auc_on_batch(masks, pred):
    # AUC 通常仅用于二值或特定分析，此处保持基础逻辑
    if pred.shape[1] > 1: return 0.5 
    aucs = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].cpu().detach().numpy().flatten()
        mask = masks[i].cpu().detach().numpy().flatten()
        if len(np.unique(mask)) > 1:
            aucs.append(roc_auc_score(mask, prediction))
    return np.mean(aucs) if aucs else 0.5

# =============================================================
# 5. 学习率调度器 (保持原样)
# =============================================================

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(f"param 'initial_lr' not specified in group {i}")
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch
        def with_counter(method):
            if getattr(method, '_with_counter', False): return method
            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                return func.__get__(instance, cls)(*args, **kwargs)
            wrapper._with_counter = True
            return wrapper
        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self, epoch=None):
        self._step_count += 1
        if epoch is None:
            self.last_epoch += 1
            values = self.get_lr()
        else:
            self.last_epoch = epoch
            values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0, self.T_i, self.T_mult, self.eta_min = T_0, T_0, T_mult, eta_min
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        self.T_cur = self.last_epoch

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1: self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i, self.T_cur = self.T_0, epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]