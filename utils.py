import os
import sys
import json
import pickle
import random
import math

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.25):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    # 统计训练集和验证集中各类别样本数
    from collections import Counter
    train_counter = Counter(train_images_label)
    val_counter   = Counter(val_images_label)
    idx_to_class  = {v: k for k, v in class_indices.items()}
    print("训练集类别分布:")
    for idx in sorted(train_counter):
        print(f"  {idx_to_class[idx]}: {train_counter[idx]}")
    print("验证集类别分布:")
    for idx in sorted(val_counter):
        print(f"  {idx_to_class[idx]}: {val_counter[idx]}")

    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, use_multimodal=False, use_twin=False, class_weights=None):
    model.train()
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        if use_twin:
            # Twin network: same three modalities fed to both sub-networks
            original, wavelet, fourier, labels = data
            sample_num += original.shape[0]
            original = original.to(device)
            wavelet = wavelet.to(device)
            fourier = fourier.to(device)
            labels = labels.to(device)
            pred = model(original, wavelet, fourier, original, wavelet, fourier)
        elif use_multimodal:
            # Multi-modal: data is (original, wavelet, fourier, labels)
            original, wavelet, fourier, labels = data
            sample_num += original.shape[0]
            
            # Move all inputs to device
            original = original.to(device)
            wavelet = wavelet.to(device)
            fourier = fourier.to(device)
            labels = labels.to(device)
            
            # Forward pass with three modalities
            try:
                pred = model(original, wavelet, fourier)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU out of memory during training. Try reducing batch_size or "
                    "enabling gradient checkpointing."
                )
        else:
            # Single-modal: data is (images, labels)
            images, labels = data
            sample_num += images.shape[0]
            pred = model(images.to(device))
            labels = labels.to(device)
        
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        # 构建进度条描述信息
        desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )
        
        # 如果是多模态且使用weighted_sum融合，显示融合权重
        if use_twin and hasattr(model, 'convnext') and hasattr(model, 'swin'):
            # TwinNetwork：分别显示 ConvNeXt 和 Swin 两个子网络的内层融合权重
            with torch.no_grad():
                if hasattr(model.convnext, 'fusion_module') and \
                        hasattr(model.convnext.fusion_module, 'raw_weights') and \
                        model.convnext.fusion_module.fusion_type == 'weighted_sum':
                    wc = torch.softmax(model.convnext.fusion_module.raw_weights, dim=0)
                    desc += ", c:[{:.2f},{:.2f},{:.2f}]".format(
                        wc[0].item(), wc[1].item(), wc[2].item())
                if hasattr(model.swin, 'fusion_module') and \
                        hasattr(model.swin.fusion_module, 'raw_weights') and \
                        model.swin.fusion_module.fusion_type == 'weighted_sum':
                    ws = torch.softmax(model.swin.fusion_module.raw_weights, dim=0)
                    desc += ", s:[{:.2f},{:.2f},{:.2f}]".format(
                        ws[0].item(), ws[1].item(), ws[2].item())
                # 显示外层门控的 bias softmax（反映两骨干的基准融合比例）
                if hasattr(model, 'twin_fusion') and hasattr(model.twin_fusion, 'gate'):
                    gate_bias = torch.softmax(model.twin_fusion.gate.bias, dim=0)
                    desc += ", g:[c={:.2f},s={:.2f}]".format(
                        gate_bias[0].item(), gate_bias[1].item())
        elif use_multimodal and hasattr(model, 'fusion_module'):
            if hasattr(model.fusion_module, 'raw_weights') and model.fusion_module.fusion_type == 'weighted_sum':
                with torch.no_grad():
                    weights = torch.softmax(model.fusion_module.raw_weights, dim=0)
                    desc += ", w:[{:.2f},{:.2f},{:.2f}]".format(
                        weights[0].item(), weights[1].item(), weights[2].item()
                    )
        
        data_loader.desc = desc

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, use_multimodal=False, use_twin=False, class_weights=None):
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        if use_twin:
            original, wavelet, fourier, labels = data
            sample_num += original.shape[0]
            original = original.to(device)
            wavelet = wavelet.to(device)
            fourier = fourier.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                pred = model(original, wavelet, fourier, original, wavelet, fourier)
        elif use_multimodal:
            # Multi-modal: data is (original, wavelet, fourier, labels)
            original, wavelet, fourier, labels = data
            sample_num += original.shape[0]
            
            # Move all inputs to device
            original = original.to(device)
            wavelet = wavelet.to(device)
            fourier = fourier.to(device)
            labels = labels.to(device)
            
            # Forward pass with three modalities
            try:
                pred = model(original, wavelet, fourier)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU out of memory during evaluation. Try reducing batch_size."
                )
        else:
            # Single-modal: data is (images, labels)
            images, labels = data
            sample_num += images.shape[0]
            pred = model(images.to(device))
            labels = labels.to(device)
        
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        # 构建进度条描述信息
        desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
        
        # 如果是多模态且使用weighted_sum融合，显示融合权重
        if use_twin and hasattr(model, 'convnext') and hasattr(model, 'swin'):
            if hasattr(model.convnext, 'fusion_module') and \
                    hasattr(model.convnext.fusion_module, 'raw_weights') and \
                    model.convnext.fusion_module.fusion_type == 'weighted_sum':
                wc = torch.softmax(model.convnext.fusion_module.raw_weights, dim=0)
                desc += ", c:[{:.2f},{:.2f},{:.2f}]".format(
                    wc[0].item(), wc[1].item(), wc[2].item())
            if hasattr(model.swin, 'fusion_module') and \
                    hasattr(model.swin.fusion_module, 'raw_weights') and \
                    model.swin.fusion_module.fusion_type == 'weighted_sum':
                ws = torch.softmax(model.swin.fusion_module.raw_weights, dim=0)
                desc += ", s:[{:.2f},{:.2f},{:.2f}]".format(
                    ws[0].item(), ws[1].item(), ws[2].item())
            # 显示外层门控的 bias softmax
            if hasattr(model, 'twin_fusion') and hasattr(model.twin_fusion, 'gate'):
                gate_bias = torch.softmax(model.twin_fusion.gate.bias, dim=0)
                desc += ", g:[c={:.2f},s={:.2f}]".format(
                    gate_bias[0].item(), gate_bias[1].item())
        elif use_multimodal and hasattr(model, 'fusion_module'):
            if hasattr(model.fusion_module, 'raw_weights') and model.fusion_module.fusion_type == 'weighted_sum':
                weights = torch.softmax(model.fusion_module.raw_weights, dim=0)
                desc += ", w:[{:.2f},{:.2f},{:.2f}]".format(
                    weights[0].item(), weights[1].item(), weights[2].item()
                )
        
        data_loader.desc = desc

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


# ============================================================================
# Multi-Modal Utility Functions
# ============================================================================

def build_multimodal_paths(original_root: str, wavelet_root: str, fourier_root: str):
    """
    从三个模态的根目录构建对应的路径列表
    
    Args:
        original_root: 原始图像根目录
        wavelet_root: 小波图像根目录
        fourier_root: 傅里叶图像根目录
        
    Returns:
        tuple: (original_paths, wavelet_paths, fourier_paths, labels)
        
    Raises:
        ValueError: 当目录结构不一致时
    """
    import os
    
    # 验证目录存在
    for root, name in [(original_root, 'original'), 
                       (wavelet_root, 'wavelet'), 
                       (fourier_root, 'fourier')]:
        if not os.path.exists(root):
            raise ValueError(f"{name} root directory does not exist: {root}")
    
    # 获取类别列表
    original_classes = sorted([d for d in os.listdir(original_root) 
                              if os.path.isdir(os.path.join(original_root, d))])
    wavelet_classes = sorted([d for d in os.listdir(wavelet_root) 
                             if os.path.isdir(os.path.join(wavelet_root, d))])
    fourier_classes = sorted([d for d in os.listdir(fourier_root) 
                             if os.path.isdir(os.path.join(fourier_root, d))])
    
    # 验证类别一致性
    if not (original_classes == wavelet_classes == fourier_classes):
        raise ValueError(
            f"Class directories mismatch:\n"
            f"  Original: {original_classes}\n"
            f"  Wavelet: {wavelet_classes}\n"
            f"  Fourier: {fourier_classes}"
        )
    
    original_paths = []
    wavelet_paths = []
    fourier_paths = []
    labels = []
    
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    
    # 遍历每个类别
    for class_idx, class_name in enumerate(original_classes):
        # 获取每个模态的图像文件
        original_class_dir = os.path.join(original_root, class_name)
        wavelet_class_dir = os.path.join(wavelet_root, class_name)
        fourier_class_dir = os.path.join(fourier_root, class_name)
        
        original_files = sorted([f for f in os.listdir(original_class_dir)
                                if os.path.splitext(f)[-1] in supported])
        wavelet_files = sorted([f for f in os.listdir(wavelet_class_dir)
                               if os.path.splitext(f)[-1] in supported])
        fourier_files = sorted([f for f in os.listdir(fourier_class_dir)
                               if os.path.splitext(f)[-1] in supported])
        
        # 验证文件名一致性
        if not (original_files == wavelet_files == fourier_files):
            raise ValueError(
                f"File mismatch in class '{class_name}':\n"
                f"  Original: {len(original_files)} files\n"
                f"  Wavelet: {len(wavelet_files)} files\n"
                f"  Fourier: {len(fourier_files)} files"
            )
        
        # 添加路径和标签
        for filename in original_files:
            original_paths.append(os.path.join(original_class_dir, filename))
            wavelet_paths.append(os.path.join(wavelet_class_dir, filename))
            fourier_paths.append(os.path.join(fourier_class_dir, filename))
            labels.append(class_idx)
    
    print(f"Found {len(original_classes)} classes with {len(original_paths)} images total")
    
    return original_paths, wavelet_paths, fourier_paths, labels


def verify_multimodal_structure(original_root: str, wavelet_root: str, fourier_root: str) -> bool:
    """
    验证三个模态的目录结构是否一致
    
    Args:
        original_root: 原始图像根目录
        wavelet_root: 小波图像根目录
        fourier_root: 傅里叶图像根目录
        
    Returns:
        bool: 结构一致返回True，否则返回False
    """
    try:
        build_multimodal_paths(original_root, wavelet_root, fourier_root)
        return True
    except (ValueError, FileNotFoundError):
        return False


def get_multimodal_dataset_info(original_root: str, wavelet_root: str, fourier_root: str) -> dict:
    """
    获取多模态数据集的统计信息
    
    Args:
        original_root: 原始图像根目录
        wavelet_root: 小波图像根目录
        fourier_root: 傅里叶图像根目录
        
    Returns:
        dict: 包含数据集统计信息的字典
    """
    import os
    
    original_paths, wavelet_paths, fourier_paths, labels = build_multimodal_paths(
        original_root, wavelet_root, fourier_root
    )
    
    # 统计每个类别的样本数
    from collections import Counter
    label_counts = Counter(labels)
    
    # 获取类别名称
    class_names = sorted([d for d in os.listdir(original_root) 
                         if os.path.isdir(os.path.join(original_root, d))])
    
    info = {
        'num_classes': len(class_names),
        'num_samples': len(original_paths),
        'class_names': class_names,
        'samples_per_class': {class_names[i]: label_counts[i] for i in range(len(class_names))},
        'original_root': original_root,
        'wavelet_root': wavelet_root,
        'fourier_root': fourier_root
    }
    
    return info


# ============================================================================
# Model Checkpoint Utility Functions
# ============================================================================

def save_multimodal_checkpoint(model, optimizer, epoch, best_acc, config, filepath: str):
    """
    保存多模态模型检查点
    
    Args:
        model: MultiModalConvNeXt模型实例
        optimizer: 优化器实例
        epoch: 当前训练轮次
        best_acc: 最佳验证准确率
        config: MultiModalConfig配置对象或字典
        filepath: 检查点保存路径
    """
    import torch
    import os
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # 准备配置字典
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError("config must be MultiModalConfig instance or dict")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        'config': config_dict
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_multimodal_checkpoint(filepath: str, model=None, optimizer=None, device='cpu'):
    """
    加载多模态模型检查点
    
    Args:
        filepath: 检查点文件路径
        model: MultiModalConvNeXt模型实例（可选）
        optimizer: 优化器实例（可选）
        device: 设备类型 ('cpu' 或 'cuda')
        
    Returns:
        dict: 包含checkpoint信息的字典，包括:
            - epoch: 训练轮次
            - best_acc: 最佳准确率
            - config: 模型配置
            如果提供了model和optimizer，它们会被就地更新
            
    Raises:
        FileNotFoundError: 当检查点文件不存在时
        RuntimeError: 当加载失败时
    """
    import torch
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {filepath}: {str(e)}") from e
    
    # 加载模型权重
    if model is not None:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model weights loaded from {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}") from e
    
    # 加载优化器状态
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Optimizer state loaded from {filepath}")
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {str(e)}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_acc': checkpoint.get('best_acc', 0.0),
        'config': checkpoint.get('config', {})
    }


def save_model_weights(model, filepath: str):
    """
    仅保存模型权重（不包含优化器和训练状态）
    
    Args:
        model: MultiModalConvNeXt模型实例
        filepath: 权重文件保存路径
    """
    import torch
    import os
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to {filepath}")


def load_model_weights(model, filepath: str, strict: bool = True, device='cpu'):
    """
    加载模型权重
    
    Args:
        model: MultiModalConvNeXt模型实例
        filepath: 权重文件路径
        strict: 是否严格匹配所有键
        device: 设备类型
        
    Raises:
        FileNotFoundError: 当权重文件不存在时
        RuntimeError: 当加载失败时
    """
    import torch
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")
    
    try:
        state_dict = torch.load(filepath, map_location=device)
        model.load_state_dict(state_dict, strict=strict)
        print(f"Model weights loaded from {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {filepath}: {str(e)}") from e


# ============================================================================
# Configuration Validation Utility Functions
# ============================================================================

def validate_multimodal_config(config) -> bool:
    """
    验证多模态配置的有效性
    
    Args:
        config: MultiModalConfig实例或配置字典
        
    Returns:
        bool: 配置有效返回True
        
    Raises:
        ValueError: 当配置无效时
    """
    from config import MultiModalConfig
    
    # 如果是字典，尝试创建MultiModalConfig实例（会自动验证）
    if isinstance(config, dict):
        try:
            MultiModalConfig.from_dict(config)
            return True
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}") from e
    
    # 如果是MultiModalConfig实例，调用其验证方法
    elif hasattr(config, '_validate'):
        try:
            config._validate()
            return True
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}") from e
    
    else:
        raise ValueError("config must be MultiModalConfig instance or dict")


def create_default_config(num_classes: int, **kwargs) -> 'MultiModalConfig':
    """
    创建默认的多模态配置
    
    Args:
        num_classes: 分类类别数
        **kwargs: 其他可选配置参数
        
    Returns:
        MultiModalConfig: 配置实例
    """
    from config import MultiModalConfig
    
    config_dict = {
        'num_classes': num_classes,
        'depths': kwargs.get('depths', [3, 3, 27, 3]),
        'dims': kwargs.get('dims', [96, 192, 384, 768]),
        'fusion_type': kwargs.get('fusion_type', 'weighted_sum'),
        'shared_weights': kwargs.get('shared_weights', False),
        'drop_path_rate': kwargs.get('drop_path_rate', 0.0),
        'wavelet_type': kwargs.get('wavelet_type', 'db1'),
        'fourier_mode': kwargs.get('fourier_mode', 'magnitude'),
        'learnable_fusion': kwargs.get('learnable_fusion', True)
    }
    
    return MultiModalConfig(**config_dict)


def print_config_summary(config):
    """
    打印配置摘要信息
    
    Args:
        config: MultiModalConfig实例或配置字典
    """
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError("config must be MultiModalConfig instance or dict")
    
    print("\n" + "="*60)
    print("Multi-Modal ConvNeXt Configuration")
    print("="*60)
    print(f"Number of classes:     {config_dict['num_classes']}")
    print(f"Architecture depths:   {config_dict['depths']}")
    print(f"Architecture dims:     {config_dict['dims']}")
    print(f"Fusion type:           {config_dict['fusion_type']}")
    print(f"Shared weights:        {config_dict['shared_weights']}")
    print(f"Drop path rate:        {config_dict['drop_path_rate']}")
    print(f"Wavelet type:          {config_dict['wavelet_type']}")
    print(f"Fourier mode:          {config_dict['fourier_mode']}")
    print(f"Learnable fusion:      {config_dict['learnable_fusion']}")
    print("="*60 + "\n")


def estimate_model_size(config) -> dict:
    """
    估算模型的参数量和内存占用
    
    Args:
        config: MultiModalConfig实例或配置字典
        
    Returns:
        dict: 包含参数量和内存估算的字典
    """
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError("config must be MultiModalConfig instance or dict")
    
    # 简化的参数量估算（基于ConvNeXt架构）
    dims = config_dict['dims']
    depths = config_dict['depths']
    num_classes = config_dict['num_classes']
    shared_weights = config_dict['shared_weights']
    
    # 每个ConvNeXt block的大约参数量（简化估算）
    params_per_block = sum(d * d * 4 for d in dims)  # 简化估算
    total_blocks = sum(depths)
    
    # 特征提取器参数量
    feature_extractor_params = params_per_block * total_blocks
    
    # 如果不共享权重，需要3个特征提取器
    if not shared_weights:
        total_params = feature_extractor_params * 3
    else:
        total_params = feature_extractor_params
    
    # 融合模块参数量（很小）
    fusion_params = 3  # 三个权重参数
    
    # 分类头参数量
    classifier_params = dims[-1] * num_classes
    
    total_params += fusion_params + classifier_params
    
    # 估算内存占用（假设float32，每个参数4字节）
    memory_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'feature_extractor_params': feature_extractor_params * (1 if shared_weights else 3),
        'fusion_params': fusion_params,
        'classifier_params': classifier_params,
        'estimated_memory_mb': memory_mb
    }
