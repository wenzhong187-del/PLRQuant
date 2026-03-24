# -*- coding: utf-8 -*-
"""
Training and Validation Loop for PLRQuant
Handles temporal sequence input [B, T, C, H, W] and computes metrics.
"""

import torch
import os
import time
import logging
import Config as config
from utils import save_on_batch


def print_summary(epoch, i, nb_batch, loss, fold, total_folds,
                  avg_loss, avg_dice, mode, lr):
    """打印训练/验证进度摘要"""
    summary = f'   [{mode}] Fold: [{fold}/{total_folds}] Epoch: [{epoch}][{i}/{nb_batch}] '
    stats = f'Loss: {loss:.4f} (Avg: {avg_loss:.4f}) | Dice: {avg_dice:.4f} '
    if mode == 'Train':
        stats += f'| LR: {lr:.2e}'
    logging.info(summary + stats)


def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, fold, total_folds):
    """
    单个 Epoch 的训练/验证逻辑
    """

    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    loss_sum, dice_sum = 0.0, 0.0
    num_samples = 0

    for i, (sampled_batch, names) in enumerate(loader, 1):

        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()


        preds = model(images)


        if config.n_labels == 1:
            loss = criterion(preds, masks.float().unsqueeze(1))
            batch_dice = criterion._show_dice(preds, masks.float().unsqueeze(1))
        else:

            loss = criterion(preds, masks.long())
            batch_dice = criterion._show_dice(preds, masks.long())


        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        current_batch_size = images.size(0)
        loss_sum += loss.item() * current_batch_size
        dice_sum += batch_dice.item() * current_batch_size
        num_samples += current_batch_size

        avg_loss = loss_sum / num_samples
        avg_dice = dice_sum / num_samples


        if logging_mode == 'Val' and (epoch + 1) % config.vis_frequency == 0:

            vis_dir = os.path.join(config.vis_path, f"fold_{fold}", f"epoch_{epoch + 1}")
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir, exist_ok=True)


            save_on_batch(images[:, config.seq_len // 2], masks.unsqueeze(1), preds, names, vis_dir)


        if i % config.print_frequency == 0 or i == len(loader):
            current_lr = optimizer.param_groups[0]['lr']
            print_summary(epoch + 1, i, len(loader), loss.item(), fold, total_folds,
                          avg_loss, avg_dice, logging_mode, current_lr)


        if config.tensorboard and writer is not None:
            step = epoch * len(loader) + i
            writer.add_scalar(f'{logging_mode}/Loss', loss.item(), step)
            writer.add_scalar(f'{logging_mode}/Dice', batch_dice.item(), step)


    if lr_scheduler is not None and logging_mode == 'Val':
        lr_scheduler.step()

    return avg_loss, avg_dice