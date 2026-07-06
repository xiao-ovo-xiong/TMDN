# Twin Multi-Domain Network (TMDN)

---

## 目录结构

```
TMDN/
├── train.py                          # 主训练脚本
├── predict.py                        # 主推理脚本
├── twin_network.py                   # TMDN 主体网络
├── multimodal_convnext.py            # 多域 ConvNeXt 子网络
├── fusion_module.py                  # 内层域间融合模块
├── model.py                          # ConvNeXt-Small 基础模型
├── multimodal_dataset.py             # 三域数据集加载
├── my_dataset.py                     # 单域数据集加载
├── image_preprocessor.py             # 小波/傅里叶变换图像生成
├── preprocess_dataset.py             # 数据集预处理入口
├── utils.py                          
├── config.py                         
├── class_indices.json                # 类别索引（训练开始时自动生成）
├── convnext_small_1k_224_ema.pth     # ConvNeXt-Small 预训练权重
└── swin_transformer/
    ├── train.py                      # Swin 侧训练脚本
    ├── predict.py                    # Swin 侧推理脚本
    ├── multimodal_swin.py            # 多域 Swin Transformer 子网络
    ├── model.py                      # Swin-Tiny 基础模型
    ├── utils.py                      
    ├── my_dataset.py                 
    ├── create_confusion_matrix.py    # 混淆矩阵生成
    ├── select_incorrect_samples.py   
    └── swin_tiny_patch4_window7_224.pth  # Swin-Tiny 预训练权重
```

---

## 数据集准备

数据集目录结构如下：

```
_DATA/
├── original/train/{false,true}/      # 原始 RGB 图像
├── wavelet/train/{false,true}/       # 小波变换图像（预处理生成）
└── fourier/train/{false,true}/       # 傅里叶幅度谱图像（预处理生成）
```

**步骤一：生成小波与傅里叶域图像**

```bash
python preprocess_dataset.py --original_root "../_DATA" --output_root "../_DATA"
```

如需处理 LCM-LoRA 生成的额外样本：

```bash
python preprocess_dataset.py --original_root ../_DATA_32 --output_root ../_DATA_32/processed
```

---

## 训练

### 1. TMDN

```bash
# 标准训练
python train.py --use-twin

# 加入 LCM-LoRA 生成式数据增强
python train.py --use-twin --extra-train-dir ../_DATA_32/original
```

### 2. 多域 ConvNeXt 子网络

```bash
python train.py --use-multimodal
```

### 3. 多域 Swin Transformer 子网络

```bash
python swin_transformer/train.py --use-multimodal
```

### 4. 各子网络单域

```bash
# ConvNeXt 单域
python train.py --data-path ../_DATA/original/train
python train.py --data-path ../_DATA/wavelet/train
python train.py --data-path ../_DATA/fourier/train

# Swin 单域
python swin_transformer/train.py --data-path ../_DATA/original/train
python swin_transformer/train.py --data-path ../_DATA/wavelet/train
python swin_transformer/train.py --data-path ../_DATA/fourier/train
```

---

## 推理

### 1. TMDN

```bash
python predict.py --mode twin --weights ./weights/best_model.pth
```

### 2. 多域 ConvNeXt 子网络

```bash
python predict.py --weights ./weights/best_model.pth
```

### 3. 多域 Swin Transformer 子网络

```bash
python swin_transformer/predict.py --weights ./weights/best_model.pth
```

### 4. 各子网络单域

```bash
# ConvNeXt 单域
python predict.py --save-confusion-matrix --modality original
python predict.py --save-confusion-matrix --modality wavelet
python predict.py --save-confusion-matrix --modality fourier

# Swin 单域
python swin_transformer/predict.py --modality original
python swin_transformer/predict.py --modality wavelet
python swin_transformer/predict.py --modality fourier
```

---
## 常用参数说明

| 参数 | 说明 |
|---|---|
| `--extra-train-dir` | 额外训练样本目录（仅追加至训练集，验证/测试集不变） |
| `--weights` | 指定推理权重路径，默认 `./weights/best_model.pth` |
| `--save-confusion-matrix` | 推理时保存混淆矩阵 |

---

## 预训练权重下载

| 权重文件 | 来源 |
|---|---|
| `convnext_small_1k_224_ema.pth` | [ConvNeXt 官方](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| `swin_transformer/swin_tiny_patch4_window7_224.pth` | [Swin Transformer 官方](https://github.com/microsoft/Swin-Transformer) |
