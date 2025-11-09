# Transformer 机器翻译项目

本项目基于 PyTorch 实现了 Transformer 架构的神经机器翻译模型，并在 IWSLT 2017 英德翻译数据集上进行了完整的实验验证。

## 项目特点
- Transformer 完整实现
- 完整的训练流程和实验分析
- 位置编码消融实验
- 多种评估指标
- 模块化代码设计，易于扩展

## 环境要求
### 基础环境
- Python 3.9
- PyTorch 2.5.1 + CUDA 12.1

### 安装依赖
bash
pip install -r requirements.txt

## 项目结构
llm/
├── train.py # 训练脚本
├── requirements.txt # 环境依赖
├── data/ # 数据目录
│ └── en-de/ # 英德翻译数据
├── results/ # 实验结果
│ ├── loss_curve.png # 损失曲线图
│ ├── accuracy_curve.png # 准确率曲线图
│ └── ablation_metrics.png # 消融实验图表
├── scripts/ # 脚本目录
│ └── run.sh # 运行指令
├── saved_models/ # 模型保存目录
│ └── transformer_*/ # 时间戳命名的模型文件夹
└── src/ # 源代码模块
├── data.py # 数据加载与预处理
├── data_download.py # 数据下载脚本
└── model.py # Transformer模型

## 快速开始
### 1. 数据准备
bash
python src/data_download.py # 下载数据
### 2. 模型训练
bash
python train.py --data_dir data/en-de --mode train --seed 42 --batch_size 64 --learning_rate 0.0003 --num_epochs 20 --d_model 128 --n_layers 2 --n_heads 4 --max_seq_length 50 --dropout 0.1
### 3. 消融实验
bash
python train.py --data_dir data/en-de --mode ablation --seed 42 --batch_size 64 --learning_rate 0.0003 --num_epochs 10 --d_model 128 --n_layers 2 --n_heads 4 --max_seq_length 50
## 实验结果
### 主要性能指标
- 最佳验证准确率: 53.45%
- 最低验证损失: 2.79
- 最佳困惑度: 16.27

### 消融实验结果
位置编码对模型性能有显著影响：
- 有位置编码: 准确率 54.12%，困惑度 15.69
- 无位置编码: 准确率 54.04%，困惑度 16.01

## 硬件要求
- GPU: NVIDIA GeForce RTX 4060（8GB）或更高
- 内存: 16GB
- 训练时间: 约2小时（20轮训练）

## 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| d_model | 模型维度 | 128 |
| n_layers | 编码器/解码器层数 | 2 |
| n_heads | 注意力头数 | 4 |
| d_ff | 前馈网络维度 | 512 |
| max_seq_length | 最大序列长度 | 50 |
| batch_size | 批大小 | 64 |
| learning_rate | 学习率 | 0.0003 |

## 复现说明
所有实验使用随机种子 42 确保可重复性。训练完成后，模型和实验结果会自动保存在 saved_models/ 目录中。
