# -*- coding: utf-8 -*-
import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from nets.PLRQuant import UCTransNet  # 修正：匹配带 D 的文件名
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from utils import dice_on_batch, iou_on_batch


def main():
    # 1. 强制覆盖 Config 里的 session 路径，确保指向 10h19
    # 只要你在 Config.py 末尾改了 test_session，这里就会自动生效
    real_session = config.test_session

    # 重新拼接路径，确保不是找“当前时间”的文件夹，而是找“测试指定”的文件夹
    config.save_path = config.task_name + '/' + config.model_name + '/' + real_session + '/'
    config.model_path = os.path.join(config.save_path, 'models/')

    print(f"--- 检查点 ---")
    print(f"预期任务文件夹: {real_session}")
    print(f"实际去找的路径: {config.model_path}")
    test_path = config.test_dataset
    if config.task_name == "MoNuSeg":  # 修正：is 改为 ==
        test_path = "./datasets/MoNuSeg/Test_Folder/"
    elif config.task_name == "pupil":
        test_path = "./datasets/pupil/Test_Folder/"
    else:
        # 默认使用 Config 中定义的测试路径
        test_path = config.test_dataset

    # 创建保存测试结果的文件夹
    save_res_path = os.path.join(config.save_path, 'test_results')
    if not os.path.exists(save_res_path):
        os.makedirs(save_res_path)

    # 2. 加载数据
    test_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(test_path, test_tf, image_size=config.img_size)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,  # 测试通常建议 batch=1 方便逐张保存
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    # 3. 初始化模型并加载权重
    config_vit = config.get_CTranS_config()
    model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    model = model.cuda()

    # 自动获取最佳模型权重路径
    load_path = os.path.join(config.model_path, f'best_model-{config.model_name}.pth.tar')
    print(f"正在加载模型权重: {load_path}")

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到权重文件，请检查 Config.py 中的 test_session 是否填写正确！")

    checkpoint = torch.load(load_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 4. 开始测试
    print("开始推理测试集...")
    dice_acc = 0.0
    iou_acc = 0.0

    with torch.no_grad():
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            images, masks = sampled_batch['image'], sampled_batch['label']
            images, masks = images.cuda(), masks.cuda()

            # --- 【创新点同步】传入 task_id ---
            task_ids = torch.full((images.size(0),), config.task_id).long().cuda()

            # 推理输出
            preds = model(images, task_id=task_ids)

            # --- 处理输出结果 ---
            if config.n_labels == 1:
                # 二值分割逻辑
                output_bin = (preds > 0.5).float()
            else:
                # 多类分割逻辑（OCT 7类）
                # 使用 argmax 取得分最高的类别索引
                output_bin = torch.argmax(preds, dim=1, keepdim=True).float()

            # 计算指标
            dice_acc += dice_on_batch(masks, preds)
            iou_acc += iou_on_batch(masks, preds)

            # 保存结果图片
            save_name = names[0][:-4]
            # 将 Tensor 转回 Numpy 格式
            pred_np = output_bin[0][0].cpu().numpy()
            mask_np = masks[0].cpu().numpy()

            # 为了方便肉眼观察，如果是多类，将像素值拉伸
            if config.n_labels > 1:
                vis_scale = 255 // (config.n_labels - 1)
                pred_img = (pred_np * vis_scale).astype(np.uint8)
                gt_img = (mask_np * vis_scale).astype(np.uint8)
            else:
                pred_img = (pred_np * 255).astype(np.uint8)
                gt_img = (mask_np * 255).astype(np.uint8)

            # 写入本地
            cv2.imwrite(os.path.join(save_res_path, f"{save_name}_pred.png"), pred_img)
            cv2.imwrite(os.path.join(save_res_path, f"{save_name}_gt.png"), gt_img)

            if i % 10 == 0:
                print(f"已处理 {i} 张图片...")

    # 5. 打印最终性能
    avg_dice = dice_acc / len(test_loader)
    avg_iou = iou_acc / len(test_loader)
    print("\n" + "=" * 30)
    print(f"测试完成！")
    print(f"平均 Dice: {avg_dice:.4f}")
    print(f"平均 IoU:  {avg_iou:.4f}")
    print(f"结果已保存至: {save_res_path}")
    print("=" * 30)


if __name__ == '__main__':
    main()