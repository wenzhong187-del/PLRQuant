# PLRQuant: An Automated Framework for Dynamic Pupillary Light Reflex Analysis

# !\[alt text](https://img.shields.io/badge/PyTorch-1.8.1-red.svg)

# 

# !\[alt text](https://img.shields.io/badge/License-MIT-blue.svg)

# 

# !\[alt text](https://img.shields.io/badge/Framework-UDTransNet%2BTCM-green.svg)

# 本研究建立了一个面向动态瞳孔对光反射（PLR）视频场景的自动化分割与量化分析框架 PLRQuant。通过改进 UDTransNet 架构并集成时域上下文模块（TCM），显著提高了在复杂采集条件下（人类与动物、近红外与可见光、静态与动态）PLR 动力学特征测量的稳定性与可重复性。



# 🌟 核心贡献 (Core Contributions)

# 增强型分割模型 (Enhanced UDTransNet + TCM):

# 基于 Dual Attention Transformer (DAT) 与 Decoder-guided Recalibration Attention (DRA) 实现跨尺度特征对齐。

# 集成了时域上下文模块 (Temporal Context Module, TCM)，利用连续帧的时序信息提升复杂干扰（如睫毛遮挡、光照突变）下的边界识别鲁棒性与帧间稳定性。

# 端到端量化流程 (End-to-End Analysis):

# 实现了从原始视频输入、瞳孔边界自动提取，到关键动力学参数（如起始直径、收缩幅度、最大收缩速度）自动计算的全自动化分析。

# 跨物种与成像条件适用性:

# 在公开人类虹膜数据集与自建小鼠动态 PLR 数据集上均表现出极高的一致性，证明了框架在 NIR（近红外）与 VIS（可见光）成像下的通用性。

# 📂 项目结构 (Project Structure)

# code

# Text

# PLRQuant/

 ├── nets/                  # 模型核心代码

 │   ├── \_\_init\_\_.py        # 接口导出

 │   ├── plr\_quant\_model.py # PLRQuantNet (UDTransNet + TCM)

 │   ├── DAT.py             # 双重注意力 Transformer

 │   ├── udtrans\_utils.py   # DRA, Up-Block 等基础组件

 │   └── TF\_configs.py      # 模型超参数配置

 ├── datasets/              # 数据集目录

 │   └── PLR\_Data/          # 建议存放 img/ 和 labelcol/

 ├── Config.py              # 全局训练与分析配置

 ├── Load\_Dataset.py        # 视频序列数据加载器 (支持 seq\_len 采样)

 ├── train\_kfold.py         # 5折交叉验证训练脚本

 ├── test\_kfold.py          # 自动化性能评估与动力学参数提取

 ├── utils.py               # 损失函数与量化指标工具包

 └── README.md

# 🚀 快速上手 (Quick Start)

# 1\. 环境准备

# code

# Bash

git clone https://github.com/PLRQuant.git

# cd PLRQuant

pip install -r requirements.txt

# 2\. 数据准备

# 请按以下结构组织你的 PLR 视频序列数据：

# code

# Text

# datasets/PLR\_Data/

├── img/                # 连续的视频帧 (如 0001.jpg, 0002.jpg...)

└── labelcol/           # 对应的瞳孔 Mask 标签 (仅训练需要, 0为背景, 1为瞳孔)

# 3\. 模型训练 (K-Fold)

# 修改 Config.py 中的 train\_dataset 路径，然后运行：

# code

# Bash

python train\_kfold.py

# 4\. 自动化量化分析 (Testing \& Analysis)

# 训练完成后，运行测试脚本。它将自动加载 5 折模型，并生成瞳孔动力学报告 plr\_dynamics\_report.csv。

# code

# Bash

# python test\_kfold.py

# 📊 PLR 动力学参数提取 (PLR Dynamics)

# PLRQuant 不仅仅是一个分割模型，它能自动计算以下关键指标：

# 指标 (Metrics)	描述 (Description)

# Resting Diameter	瞳孔在受光刺激前的起始直径 (mm)

# Peak Constriction	瞳孔收缩后的最小直径 (mm)

# Constriction Ratio	瞳孔收缩率 (%) = (D\_max - D\_min) / D\_max

# Max Velocity (MCV)	最大收缩速度 (mm/s)，反映副交感神经活性






