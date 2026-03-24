import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from scipy.ndimage import rotate
from scipy.ndimage import zoom


class PupilTemporalTransform:
    """
    时序增强：确保序列中的所有帧执行相同的随机变换
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img_list, mask):
        # 1. 随机决定变换参数
        do_flip = random.random() > 0.5
        do_rot = random.random() > 0.5
        rot_angle = np.random.randint(-20, 20) if do_rot else 0
        flip_axis = np.random.randint(0, 2) if do_flip else -1

        processed_imgs = []
        # 2. 对所有图像应用相同变换
        for img in img_list:
            if do_rot:
                img = rotate(img, rot_angle, order=1, reshape=False)
            if do_flip:
                img = np.flip(img, axis=flip_axis).copy()

            # Resize
            h, w = img.shape[:2]
            if h != self.output_size[0] or w != self.output_size[1]:
                img = cv2.resize(img, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)

            # To Tensor [C, H, W]
            img = F.to_tensor(img)
            processed_imgs.append(img)

        # 3. 对 Mask 应用相同变换 (Mask 必须用最近邻插值)
        if do_rot:
            mask = rotate(mask, rot_angle, order=0, reshape=False)
        if do_flip:
            mask = np.flip(mask, axis=flip_axis).copy()

        mask = cv2.resize(mask, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).long()
        mask[mask > 0] = 1  # 保证二值化

        # 堆叠序列: [T, C, H, W]
        img_sequence = torch.stack(processed_imgs, dim=0)

        return img_sequence, mask


class PLRVideoDataset(Dataset):
    """
    PLR 视频序列数据集
    输入: 连续的 T 帧图像
    输出: (序列张量 [T, C, H, W], 中间帧的标签 [H, W])
    """

    def __init__(self, dataset_path, filelists, seq_len=5, image_size=224, split='train'):
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'img')
        self.mask_path = os.path.join(dataset_path, 'labelcol')
        self.filelists = sorted(filelists)  # 确保帧顺序正确
        self.seq_len = seq_len
        self.image_size = image_size
        self.split = split
        self.transform = PupilTemporalTransform(output_size=(image_size, image_size))

    def __len__(self):
        # 为了保证取到完整的序列，减去边界
        return len(self.filelists) - self.seq_len + 1

    def __getitem__(self, idx):
        # 1. 获取连续的 T 帧文件名
        seq_files = self.filelists[idx: idx + self.seq_len]

        # 2. 读取图像序列
        img_list = []
        for f in seq_files:
            img = cv2.imread(os.path.join(self.img_path, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)

        # 3. 读取中间帧的标签 (TCM 的训练目标通常是序列的中心帧)
        mid_idx = self.seq_len // 2
        target_file = seq_files[mid_idx]
        # 假设标签后缀为 .png
        mask_name = target_file.replace('.jpg', '.png').replace('.bmp', '.png')
        mask = cv2.imread(os.path.join(self.mask_path, mask_name), 0)

        # 4. 应用同步增强
        img_seq, mask_tensor = self.transform(img_list, mask)

        return {'image': img_seq, 'label': mask_tensor}, target_file


def get_file_list(path, fold_files=None):
    """
    根据路径和 K-Fold 列表获取有效文件
    """
    all_files = sorted(os.listdir(os.path.join(path, 'img')))
    if fold_files is not None:
        return [f for f in all_files if f in fold_files]
    return all_files