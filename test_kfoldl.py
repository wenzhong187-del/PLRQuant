# -*- coding: utf-8 -*-
"""
Final Testing & PLR Quantitative Analysis Script
1. Evaluate segmentation (Dice/IoU)
2. Extract PLR Dynamics (Diameter, Velocity, Constriction Ratio)
3. Export structured data for research.
"""

import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import Config as config
from Load_Dataset import PLRVideoDataset
from nets import PLRQuantNet, get_plr_quant_config
from utils import dice_coef


def calculate_plr_metrics(diameters, fps, ratio):
    """
    瞳孔动力学量化分析逻辑
    diameters: 每一帧的直径列表 (单位: pixel)
    fps: 视频帧率
    ratio: 像素到毫米的转换系数
    """
    diams_mm = np.array(diameters) * ratio


    d_max = np.max(diams_mm)
    d_min = np.min(diams_mm)

    constriction_amp = d_max - d_min
    constriction_ratio = (constriction_amp / d_max) * 100

    # 计算速度 (差分法)
    velocities = np.abs(np.diff(diams_mm)) * fps
    mcv = np.max(velocities) if len(velocities) > 0 else 0

    return {
        'Resting_Diameter_mm': round(d_max, 4),
        'Peak_Constriction_mm': round(d_min, 4),
        'Constriction_Amplitude_mm': round(constriction_amp, 4),
        'Constriction_Ratio_pct': round(constriction_ratio, 2),
        'Max_Velocity_mcv': round(mcv, 4)
    }


def test_plr_framework():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    img_dir = os.path.join(config.test_dataset, 'img')
    filelists = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.bmp'))])


    all_fold_dice = []
    final_dynamics_log = []


    for fold in range(1, config.kfold + 1):
        print(f"\n>>> Evaluating Fold {fold}...")


        model_path = os.path.join(config.model_path, f"fold_{fold}", f"best_model-{config.model_name}.pth.tar")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found at {model_path}, skipping.")
            continue

        config_vit = get_plr_quant_config()
        model = PLRQuantNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()


        test_ds = PLRVideoDataset(config.test_dataset, filelists, seq_len=config.seq_len, split='test')
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        fold_dice = []
        current_sequence_diams = []

        with torch.no_grad():
            for i, (sampled_batch, names) in enumerate(tqdm(test_loader, desc=f"Fold {fold}")):
                images, masks = sampled_batch['image'].to(device), sampled_batch['label'].to(device)


                preds = model(images)


                pred_np = (preds > 0.5).cpu().numpy().astype(np.uint8)
                mask_np = masks.cpu().numpy().astype(np.uint8)
                d_score = dice_coef(mask_np, pred_np)
                fold_dice.append(d_score)


                pixel_area = np.sum(pred_np > 0)
                # 等效直径计算 D = 2 * sqrt(Area / pi)
                pixel_diameter = 2 * np.sqrt(pixel_area / np.pi)
                current_sequence_diams.append(pixel_diameter)


                if fold == 1 and i % 50 == 0:
                    vis_save_path = os.path.join(config.save_path, 'test_visualize')
                    os.makedirs(vis_save_path, exist_ok=True)

                    raw_img = images[0, config.seq_len // 2].cpu().permute(1, 2, 0).numpy() * 255
                    raw_img = cv2.cvtColor(raw_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    contours, _ = cv2.findContours(pred_np[0, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(raw_img, contours, -1, (0, 255, 0), 2)  # 绿色为预测边界
                    cv2.imwrite(os.path.join(vis_save_path, f"res_{names[0]}"), raw_img)


        avg_dice = np.mean(fold_dice)
        all_fold_dice.append(avg_dice)
        print(f"Fold {fold} Mean Dice: {avg_dice:.4f}")


        if fold == 1:
            dynamics = calculate_plr_metrics(current_sequence_diams, config.video_fps, config.pixel_to_mm_ratio)
            final_dynamics_log.append(dynamics)


    print("\n" + "=" * 40)
    print("Final Framework Performance:")
    print(f"Mean K-Fold Dice: {np.mean(all_fold_dice):.4f} ± {np.std(all_fold_dice):.4f}")


    if final_dynamics_log:
        df = pd.DataFrame(final_dynamics_log)
        report_path = os.path.join(config.save_path, 'plr_dynamics_report.csv')
        df.to_csv(report_path, index=False)
        print(f"PLR Dynamics Report saved to: {report_path}")
        print("Sample Metrics:", dynamics)
    print("=" * 40)


if __name__ == '__main__':
    test_plr_framework()
