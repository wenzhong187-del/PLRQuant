# -*- coding: utf-8 -*-
"""
Config for PLRQuant
Focused on Pupillary Light Reflex (PLR) Video Segmentation and Analysis.
"""

import os
import time
import argparse

task_name = 'PLRQuant'
model_name = 'PLRQuantNet'
train_dataset = './datasets/PLR_Data/' 
test_dataset  = './datasets/PLR_Data/' 

session_name = 'PLR_Session_' + time.strftime('%m.%d_%Hh%M')
save_path    = os.path.join(task_name + '_results', model_name, session_name)
model_path   = os.path.join(save_path, 'models/')
log_path     = os.path.join(save_path, session_name + ".log")
vis_path     = os.path.join(save_path, 'visualize_val/')

for p in [model_path, vis_path]:
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)
epochs = 300
batch_size = 4
learning_rate = 1e-4
early_stopping_patience = 40
# 图像与时序设置
img_size = 224
seq_len = 5
n_channels = 3
n_labels = 1
# 实验设置
kfold = 5
seed = 666
cosineLR = True
print_frequency = 10
save_frequency = 50
vis_frequency = 10
tensorboard = True
tensorboard_folder = os.path.join(save_path, 'tensorboard_logs/')
pixel_to_mm_ratio = 0.05
video_fps = 60


