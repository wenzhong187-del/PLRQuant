# -*- coding: utf-8 -*-
"""
Main Training Script for PLRQuant
Implements 5-Fold Cross-Validation for Pupil Video Segmentation.
"""

import os
import torch
import torch.optim as optim
from torch.backends import cudnn
from tensorboardX import SummaryWriter
import numpy as np
import random
import logging
import warnings
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import Config as config
from Load_Dataset import PLRVideoDataset
from nets import PLRQuantNet, get_plr_quant_config
from train_one_epoch import train_one_epoch
from utils import BinaryDiceBCE, CosineAnnealingWarmRestarts

warnings.filterwarnings("ignore")


def logger_config(log_path):
    """配置日志系统，同时输出到文件和控制台"""
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def save_checkpoint(state, save_path, fold, model_type):
    """保存模型权重"""
    fold_path = os.path.join(save_path, f"fold_{fold}")
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    filename = os.path.join(fold_path, f'best_model-{model_type}.pth.tar')
    torch.save(state, filename)
    logging.info(f'\t [Checkpoint] Saved to {filename}')


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def main_loop(train_loader, val_loader, fold, total_folds, writer):
    """单折训练主循环"""
    # 1. 初始化模型
    config_vit = get_plr_quant_config()
    model = PLRQuantNet(
        config_vit,
        n_channels=config.n_channels,
        n_classes=config.n_labels,
        img_size=config.img_size
    ).cuda()

    # 2. 损失函数与优化器
    # 瞳孔分割是二分类，使用 BinaryDiceBCE 综合 Loss
    criterion = BinaryDiceBCE(dice_weight=1, BCE_weight=1)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 3. 学习率调度 (余弦退火)
    if config.cosineLR:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    else:
        lr_scheduler = None

    max_dice = 0.0
    best_epoch = 1

    for epoch in range(config.epochs):
        logging.info(f'\n--- Fold [{fold}/{total_folds}] | Epoch [{epoch + 1}/{config.epochs}] ---')

        # 训练阶段
        model.train()
        train_loss, train_dice = train_one_epoch(
            train_loader, model, criterion, optimizer, writer, epoch, None, fold, total_folds
        )

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss, val_dice = train_one_epoch(
                val_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, fold, total_folds
            )

        # 保存最优模型
        if val_dice > max_dice:
            max_dice = val_dice
            best_epoch = epoch + 1
            save_checkpoint({
                'epoch': epoch + 1,
                'model': config.model_name,
                'state_dict': model.state_dict(),
                'val_dice': val_dice,
                'optimizer': optimizer.state_dict()
            }, config.model_path, fold, config.model_name)

        logging.info(f'\t [Val] Current Dice: {val_dice:.4f} | Best Dice: {max_dice:.4f} at Epoch {best_epoch}')

        # 早停判断
        if epoch + 1 - best_epoch > config.early_stopping_patience:
            logging.info(f'\t [Early Stopping] Triggered at epoch {epoch + 1}')
            break

    return max_dice


if __name__ == '__main__':

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True


    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    logger = logger_config(log_path=config.log_path)



    img_dir = os.path.join(config.train_dataset, 'img')
    filelists = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.bmp'))])
    filelists = np.array(filelists)


    kf = KFold(n_splits=config.kfold, shuffle=True, random_state=config.seed)
    fold_results = []


    writer = SummaryWriter(config.tensorboard_folder) if config.tensorboard else None

    for fold, (train_index, val_index) in enumerate(kf.split(filelists), 1):
        train_files = filelists[train_index]
        val_files = filelists[val_index]

        logging.info(f'\n{"=" * 30}\n Starting Fold {fold}/{config.kfold} \n{"=" * 30}')
        logging.info(f'Train samples: {len(train_files)}, Val samples: {len(val_files)}')

        # 初始化视频序列数据集 (TCM 核心)
        train_ds = PLRVideoDataset(config.train_dataset, train_files, seq_len=config.seq_len)
        val_ds = PLRVideoDataset(config.train_dataset, val_files, seq_len=config.seq_len)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)


        best_dice = main_loop(train_loader, val_loader, fold, config.kfold, writer)
        fold_results.append(best_dice)


    avg_dice = np.mean(fold_results)
    std_dice = np.std(fold_results)
    logging.info(f'\n{"=" * 30}\n Final K-Fold Results: \n{"=" * 30}')
    for i, res in enumerate(fold_results):
        logging.info(f'Fold {i + 1}: {res:.4f}')
    logging.info(f'Average Dice: {avg_dice:.4f} ± {std_dice:.4f}')

    if writer:
        writer.close()
