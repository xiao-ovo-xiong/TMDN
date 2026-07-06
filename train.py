import os
import sys
import importlib
import random
import numpy as np

_root = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if not p.endswith('swin_transformer')]
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from multimodal_dataset import MultiModalDataset
from model import convnext_small as create_model
from multimodal_convnext import MultiModalConvNeXt
from twin_network import TwinNetwork

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("root_utils", os.path.join(_root, "utils.py"))
_root_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
read_split_data = _root_utils.read_split_data
create_lr_scheduler = _root_utils.create_lr_scheduler
get_params_groups = _root_utils.get_params_groups
train_one_epoch = _root_utils.train_one_epoch
evaluate = _root_utils.evaluate


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=5, delta=0.001, mode="max"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == "max":
            if current_score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
        elif self.mode == "min":
            if current_score > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0


def _inject_extra_train(train_paths: list, train_labels: list, extra_dir: str):
    """
    将 extra_dir 下的样本全部追加到训练集，不参与 train/val 划分。
    extra_dir 目录结构：extra_dir/{class_name}/*.jpg
    class_name 必须与主数据集的类别名称一致（通过 class_indices.json 映射）。
    """
    supported = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
    # 读取类别映射
    assert os.path.exists('class_indices.json'), \
        "class_indices.json not found. Run read_split_data first."
    import json
    with open('class_indices.json') as f:
        class_indict = json.load(f)  # {idx_str: class_name}
    name_to_idx = {v: int(k) for k, v in class_indict.items()}

    extra_count = 0
    for class_name in sorted(os.listdir(extra_dir)):
        class_dir = os.path.join(extra_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in name_to_idx:
            print(f"  警告: extra_train_dir 中的类别 '{class_name}' 不在 class_indices.json，跳过")
            continue
        label = name_to_idx[class_name]
        for fname in sorted(os.listdir(class_dir)):
            if os.path.splitext(fname)[1] in supported:
                fpath = os.path.join(class_dir, fname).replace('\\', '/')
                train_paths.append(fpath)
                train_labels.append(label)
                extra_count += 1

    print(f"额外训练样本注入: {extra_count} 张（来自 {extra_dir}），"
          f"训练集总计 {len(train_paths)} 张")
    return train_paths, train_labels


def set_seed(seed: int = 42):
    """固定所有随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed fixed: {args.seed}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # Determine training mode
    use_twin = getattr(args, 'use_twin', False)
    use_multimodal = args.use_multimodal

    if use_twin:
        print("Using twin network training mode")
        use_multimodal = True  # twin 模式也使用多模态数据集

    if use_multimodal:
        if not use_twin:
            print("Using multi-modal training mode")
        # For multi-modal, we need to construct paths for all three modalities
        # Assuming data structure: data_root/{original,wavelet,fourier}/train/...
        data_root = os.path.dirname(args.data_path)
        
        # Read split data from original path
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
            args.data_path)
        
        # 注入额外训练样本（仅进训练集，不参与 train/val 划分）
        if args.extra_train_dir:
            train_images_path, train_images_label = _inject_extra_train(
                train_images_path, train_images_label, args.extra_train_dir)

        # Convert original paths to wavelet and fourier paths
        train_wavelet_paths = [p.replace('/original/', '/wavelet/') for p in train_images_path]
        train_fourier_paths = [p.replace('/original/', '/fourier/') for p in train_images_path]
        val_wavelet_paths = [p.replace('/original/', '/wavelet/') for p in val_images_path]
        val_fourier_paths = [p.replace('/original/', '/fourier/') for p in val_images_path]
    else:
        print("Using single-modal training mode")
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
            args.data_path)
        
        # 注入额外训练样本（仅进训练集，不参与 train/val 划分）
        if args.extra_train_dir:
            train_images_path, train_images_label = _inject_extra_train(
                train_images_path, train_images_label, args.extra_train_dir)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 实例化训练数据集和验证数据集
    if use_multimodal:
        train_dataset = MultiModalDataset(
            original_paths=train_images_path,
            wavelet_paths=train_wavelet_paths,
            fourier_paths=train_fourier_paths,
            labels=train_images_label,
            transform=data_transform["train"]
        )
        
        val_dataset = MultiModalDataset(
            original_paths=val_images_path,
            wavelet_paths=val_wavelet_paths,
            fourier_paths=val_fourier_paths,
            labels=val_images_label,
            transform=data_transform["val"]
        )
    else:
        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=data_transform["train"])
        
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])

    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    
    # Use appropriate collate_fn based on dataset type
    train_collate_fn = train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None
    val_collate_fn = val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_collate_fn)

    # Create model based on mode
    if use_twin:
        model = TwinNetwork(
            num_classes=args.num_classes,
            convnext_depths=args.depths,
            convnext_dims=args.dims,
            convnext_fusion_type=args.fusion_type,
            swin_embed_dim=args.swin_embed_dim,
            swin_depths=tuple(args.swin_depths),
            swin_num_heads=tuple(args.swin_num_heads),
            swin_window_size=args.swin_window_size,
            twin_fusion_type=args.twin_fusion_type,
        ).to(device)
        model.load_pretrained_convnext(args.convnext_weights)
        model.load_pretrained_swin(args.swin_weights)
    elif use_multimodal:
        model = MultiModalConvNeXt(
            num_classes=args.num_classes,
            depths=args.depths,
            dims=args.dims,
            fusion_type=args.fusion_type,
            shared_weights=args.shared_weights,
            drop_path_rate=0.
        ).to(device)
        
        # Load pretrained weights if provided
        if args.weights != "":
            assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
            try:
                model.load_pretrained(args.weights, strict=False)
                print(f"Successfully loaded pretrained weights from {args.weights}")
            except Exception as e:
                print(f"Warning: Failed to load pretrained weights: {str(e)}")
                print("Continuing with random initialization...")
    else:
        model = create_model(num_classes=args.num_classes).to(device)
        
        if args.weights != "":
            assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
            weights_dict = torch.load(args.weights, map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 为融合权重设置更大的学习率
    if use_twin:
        # twin_fusion 门控参数 + 两个子网络的内层 raw_weights 均使用更高学习率
        twin_fusion_params = list(model.twin_fusion.parameters())
        inner_fusion_params = []
        if hasattr(model.convnext, 'fusion_module') and \
                hasattr(model.convnext.fusion_module, 'raw_weights'):
            inner_fusion_params.append(model.convnext.fusion_module.raw_weights)
        if hasattr(model.swin, 'fusion_module') and \
                hasattr(model.swin.fusion_module, 'raw_weights'):
            inner_fusion_params.append(model.swin.fusion_module.raw_weights)

        special_param_ids = {id(p) for p in twin_fusion_params + inner_fusion_params}
        other_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in special_param_ids]
        twin_lr = args.lr * args.twin_fusion_lr_multiplier
        optimizer = optim.AdamW([
            {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd},
            {'params': twin_fusion_params, 'lr': twin_lr, 'weight_decay': 0.0},
            {'params': inner_fusion_params, 'lr': twin_lr, 'weight_decay': 0.0},
        ])
        print(f"Base lr: {args.lr:.5f}, TwinFusion gate lr: {twin_lr:.5f}, "
              f"Inner fusion lr: {twin_lr:.5f}")
    elif use_multimodal and hasattr(model, 'fusion_module'):
        if hasattr(model.fusion_module, 'raw_weights'):
            # 创建参数组：融合权重使用自定义倍数的学习率
            fusion_params = [model.fusion_module.raw_weights]
            other_params = [p for n, p in model.named_parameters() 
                           if p.requires_grad and 'raw_weights' not in n]
            
            fusion_lr = args.lr * args.fusion_lr_multiplier
            optimizer = optim.AdamW([
                {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd},
                {'params': fusion_params, 'lr': fusion_lr, 'weight_decay': 0.0}
            ])
            print(f"Base learning rate: {args.lr:.5f}")
            print(f"Fusion weights learning rate: {fusion_lr:.5f} ({args.fusion_lr_multiplier}x base lr)")
        else:
            pg = get_params_groups(model, weight_decay=args.wd)
            optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    else:
        pg = get_params_groups(model, weight_decay=args.wd)
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    # 初始化早停
    early_stopping = EarlyStopping(patience=args.patience,
                                   delta=args.delta,
                                   mode="max")

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler,
                                                use_multimodal=use_multimodal,
                                                use_twin=use_twin,
                                                class_weights=args.class_weights)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     use_multimodal=use_multimodal,
                                     use_twin=use_twin,
                                     class_weights=args.class_weights)

        tags = ["train_loss", "train_acc",
                "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 早停判断
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(
                f"\nEarly stopping triggered at epoch {epoch}! Best acc: {best_acc:.4f}")
            break

        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/best_model.pth")
            best_acc = val_acc
            print(f"Saved best model with accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=88)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=5e-2)

    parser.add_argument('--data-path', type=str,
                        default="../_DATA_2/original/train")

    parser.add_argument('--weights', type=str, default='convnext_small_1k_224_ema.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    # 早停参数
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--delta', type=float, default=0.001,
                        help='Minimum change to qualify as improvement')
    
    # 多模态参数
    parser.add_argument('--use-multimodal', action='store_true',
                        help='Use multi-modal training with original, wavelet, and fourier images')
    parser.add_argument('--fusion-type', type=str, default='weighted_sum',
                        choices=['weighted_sum', 'concat', 'attention'],
                        help='Feature fusion strategy')
    parser.add_argument('--shared-weights', action='store_true',
                        help='Share weights across the three feature extractors')
    parser.add_argument('--fusion-lr-multiplier', type=float, default=15.0,
                        help='Learning rate multiplier for fusion weights')
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 3, 27, 3],
                        help='ConvNeXt depths for each stage')
    parser.add_argument('--dims', type=int, nargs='+', default=[96, 192, 384, 768],
                        help='ConvNeXt dimensions for each stage')

    # 双胞胎网络参数
    parser.add_argument('--use-twin', action='store_true',
                        help='Enable twin network training mode')
    parser.add_argument('--twin-fusion-type', type=str, default='prob_gating',
                        choices=['logits_gating', 'prob_gating', 'feature_gating'],
                        help='Twin fusion strategy')
    parser.add_argument('--convnext-weights', type=str, default='./convnext_small_1k_224_ema.pth',
                        help='Pretrained weights path for ConvNeXt sub-network')
    parser.add_argument('--swin-weights', type=str, default='./swin_transformer/swin_tiny_patch4_window7_224.pth',
                        help='Pretrained weights path for Swin sub-network')
    parser.add_argument('--swin-embed-dim', type=int, default=96,
                        help='Swin Transformer embed_dim')
    parser.add_argument('--swin-depths', type=int, nargs='+', default=[2, 2, 6, 2],
                        help='Swin Transformer depths for each stage')
    parser.add_argument('--swin-num-heads', type=int, nargs='+', default=[3, 6, 12, 24],
                        help='Swin Transformer num_heads for each stage')
    parser.add_argument('--swin-window-size', type=int, default=7,
                        help='Swin Transformer window size')
    parser.add_argument('--twin-fusion-lr-multiplier', type=float, default=15.0,
                        help='Learning rate multiplier for twin fusion gate parameters')
    parser.add_argument('--class-weights', type=float, nargs='+', default=None,
                        help='Class weights for CrossEntropyLoss, e.g. --class-weights 1.0 9.0 '
                             'to upweight the minority class (true). '
                             'Order matches class index (false=0, true=1).')
    parser.add_argument('--extra-train-dir', type=str, default=None,
                        help='额外训练样本目录，该目录下的所有样本全部追加到训练集，'
                             '不参与 train/val 划分。目录结构：{class_name}/*.jpg。'
                             '用于 LoRA 数据增强消融实验，保证 val/test 集不变。'
                             '例：--extra-train-dir ../_LORA_DATA/original')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility. Not set by default. '
                             'Use e.g. --seed 42 to enable.')
    opt = parser.parse_args()

    main(opt)
