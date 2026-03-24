# -*- coding: utf-8 -*-
"""
Core Inference Logic for PLRQuant
Processes video sequences and extracts segmentation metrics + pupil features.
"""

import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from utils import dice_coef, jaccard_score


def test_one_epoch(loader, model, device, config, save_vis=False, epoch=None, fold=None):
    """
    对测试集/验证集进行一轮完整的推理
    1. 计算分割指标 (Dice, IoU)
    2. 提取每帧瞳孔面积与等效直径
    """
    model.eval()

    total_dice = []
    total_iou = []
    pupil_data_list = []


    if save_vis:
        vis_dir = os.path.join(config.vis_path, f"fold_{fold}", f"test_epoch_{epoch if epoch else 'final'}")
        os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for i, (sampled_batch, names) in enumerate(tqdm(loader, desc="Inference", leave=False)):

            images, masks = sampled_batch['image'].to(device), sampled_batch['label'].to(device)


            preds = model(images)


            pred_np = (preds > 0.5).cpu().numpy().astype(np.uint8)
            mask_np = masks.cpu().numpy().astype(np.uint8)


            d_score = dice_coef(mask_np, pred_np)
            i_score = jaccard_score(mask_np.flatten(), pred_np.flatten())
            total_dice.append(d_score)
            total_iou.append(i_score)


            for b in range(pred_np.shape[0]):
                pixel_area = np.sum(pred_np[b, 0] > 0)

                pixel_diameter = 2 * np.sqrt(pixel_area / np.pi)


                mm_diameter = pixel_diameter * config.pixel_to_mm_ratio

                pupil_data_list.append({
                    'filename': names[b],
                    'area_px': pixel_area,
                    'diameter_px': round(pixel_diameter, 3),
                    'diameter_mm': round(mm_diameter, 4)
                })


            if save_vis and i % 20 == 0:

                mid_frame = images[0, config.seq_len // 2].cpu().permute(1, 2, 0).numpy()
                mid_frame = (mid_frame * 255).astype(np.uint8)
                mid_frame = cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR)


                contours, _ = cv2.findContours(pred_np[0, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mid_frame, contours, -1, (0, 255, 0), 2)


                cv2.putText(mid_frame, f"D: {mm_diameter:.2f}mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imwrite(os.path.join(vis_dir, f"res_{names[0]}"), mid_frame)

    # 汇总结果
    metrics = {
        'mean_dice': np.mean(total_dice),
        'mean_iou': np.mean(total_iou),
    }

    return metrics, pupil_data_list