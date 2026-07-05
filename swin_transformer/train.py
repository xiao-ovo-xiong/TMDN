import os
import sys
import argparse
import math
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# 将父目录加入 sys.path，以便导入根目录的模块
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from my_dataset import MyDataSet
from multimodal_dataset import MultiModalDataset
from swin_transformer.model import swin_tiny_patch4_window7_224 as create_model
from swin_transformer.multimodal_swin import MultiModalSwinTransformer
from swin_transformer.utils import read_split_data, train_one_epoch, evaluate


class EarlyStopping:
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
            return
        if self.mode == "max":
            if current_score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
        else:
            if current_score > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0


def train_one_epoch_multimodal(model, optimizer, data_loader, device, epoch, lr_scheduler=None, class_weights=None):
    model.train()
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        original, wavelet, fourier, labels = data
        sample_num += original.shape[0]
        original, wavelet, fourier = original.to(device), wavelet.to(device), fourier.to(device)
        labels = labels.to(device)
        pred = model(original, wavelet, fourier)
        accu_num += torch.eq(torch.max(pred, dim=1)[1], labels).sum()
        loss = loss_fn(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        # 构建进度条描述，格式与 ConvNeXt 保持一致
        desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )
        # 显示三模态融合权重
        if hasattr(model, 'fusion_module') and hasattr(model.fusion_module, 'raw_weights') \
                and model.fusion_module.fusion_type == 'weighted_sum':
            with torch.no_grad():
                w = torch.softmax(model.fusion_module.raw_weights, dim=0)
                desc += ", w:[{:.2f},{:.2f},{:.2f}]".format(w[0].item(), w[1].item(), w[2].item())
        data_loader.desc = desc

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_multimodal(model, data_loader, device, epoch, class_weights=None):
    model.eval()
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        original, wavelet, fourier, labels = data
        sample_num += original.shape[0]
        original, wavelet, fourier = original.to(device), wavelet.to(device), fourier.to(device)
        labels = labels.to(device)
        pred = model(original, wavelet, fourier)
        accu_num += torch.eq(torch.max(pred, dim=1)[1], labels).sum()
        loss = loss_fn(pred, labels)
        accu_loss += loss

        # 构建进度条描述，格式与 ConvNeXt 保持一致
        desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
        # 显示三模态融合权重
        if hasattr(model, 'fusion_module') and hasattr(model.fusion_module, 'raw_weights') \
                and model.fusion_module.fusion_type == 'weighted_sum':
            w = torch.softmax(model.fusion_module.raw_weights, dim=0)
            desc += ", w:[{:.2f},{:.2f},{:.2f}]".format(w[0].item(), w[1].item(), w[2].item())
        data_loader.desc = desc
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


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

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    use_multimodal = args.use_multimodal

    # 读取数据路径
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_path if not use_multimodal else os.path.join(args.data_root, "original/train")
    )

    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            # transforms.RandomHorizontalFlip(), # 禁用了水平翻转（原模型自带）
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

    if use_multimodal:
        train_wavelet = [p.replace('/original/', '/wavelet/') for p in train_images_path]
        train_fourier = [p.replace('/original/', '/fourier/') for p in train_images_path]
        val_wavelet = [p.replace('/original/', '/wavelet/') for p in val_images_path]
        val_fourier = [p.replace('/original/', '/fourier/') for p in val_images_path]
        train_dataset = MultiModalDataset(train_images_path, train_wavelet, train_fourier,
                                          train_images_label, transform=data_transform["train"])
        val_dataset = MultiModalDataset(val_images_path, val_wavelet, val_fourier,
                                        val_images_label, transform=data_transform["val"])
    else:
        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=data_transform["train"])
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    collate_fn = train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=nw, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=nw,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None)

    # 创建模型
    if use_multimodal:
        print("Using MultiModalSwinTransformer")
        model = MultiModalSwinTransformer(
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depths=tuple(args.depths),
            num_heads=tuple(args.num_heads),
            window_size=args.window_size,
            fusion_type=args.fusion_type,
        ).to(device)
        if args.weights:
            model.load_pretrained(args.weights, strict=False)
    else:
        model = create_model(num_classes=args.num_classes).to(device)
        if args.weights:
            assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
            weights_dict = torch.load(args.weights, map_location=device)["model"]
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "classifier" not in name:
                para.requires_grad_(False)

    # 优化器：多模态时融合权重使用更高学习率，与 ConvNeXt 侧保持一致
    if use_multimodal and hasattr(model, 'fusion_module') and \
            hasattr(model.fusion_module, 'raw_weights'):
        fusion_params = [model.fusion_module.raw_weights]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and 'raw_weights' not in n]
        fusion_lr = args.lr * args.fusion_lr_multiplier
        optimizer = optim.AdamW([
            {'params': other_params, 'lr': args.lr, 'weight_decay': 5e-2},
            {'params': fusion_params, 'lr': fusion_lr, 'weight_decay': 0.0},
        ])
        print(f"Base learning rate: {args.lr:.5f}")
        print(f"Fusion weights learning rate: {fusion_lr:.5f} ({args.fusion_lr_multiplier}x base lr)")
    else:
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)

    # 余弦退火 + warmup，与 ConvNeXt 侧保持一致
    def _lr_lambda(x):
        warmup_steps = 1 * len(train_loader)  # 1 epoch warmup
        total_steps = args.epochs * len(train_loader)
        if x <= warmup_steps:
            return 1e-3 * (1 - x / warmup_steps) + x / warmup_steps
        current = x - warmup_steps
        cosine_steps = total_steps - warmup_steps
        return ((1 + math.cos(current * math.pi / cosine_steps)) / 2) * (1 - 1e-6) + 1e-6

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, mode="max")
    best_acc = 0.

    for epoch in range(args.epochs):
        if use_multimodal:
            train_loss, train_acc = train_one_epoch_multimodal(
                model, optimizer, train_loader, device, epoch, lr_scheduler,
                class_weights=args.class_weights)
            val_loss, val_acc = evaluate_multimodal(
                model, val_loader, device, epoch,
                class_weights=args.class_weights)
        else:
            train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch)
            val_loss, val_acc = evaluate(model, val_loader, device, epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch}! Best acc: {best_acc:.4f}")
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
    parser.add_argument('--lr', type=float, default=5e-5) # 原始值为1e-4
    parser.add_argument('--data-path', type=str, default="../_DATA_2/original/train",
                        help='Data path for single-modal training')
    parser.add_argument('--modality', type=str, default='original',
                        choices=['original', 'wavelet', 'fourier'],
                        help='Modality to use for single-modal training')
    parser.add_argument('--weights', type=str, default='./swin_transformer/swin_tiny_patch4_window7_224.pth',
                        help='Initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')

    # 早停参数
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--delta', type=float, default=0.001)

    # 多模态参数
    parser.add_argument('--use-multimodal', action='store_true',
                        help='Enable multimodal training with MultiModalSwinTransformer')
    parser.add_argument('--fusion-type', type=str, default='weighted_sum',
                        choices=['weighted_sum', 'concat', 'attention'])
    parser.add_argument('--fusion-lr-multiplier', type=float, default=15.0,
                        help='Learning rate multiplier for fusion weights (default: 10.0)')
    parser.add_argument('--data-root', type=str, default='../_DATA_2',
                        help='Root dir containing original/wavelet/fourier subdirs')
    parser.add_argument('--embed-dim', type=int, default=96)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6, 2])
    parser.add_argument('--num-heads', type=int, nargs='+', default=[3, 6, 12, 24])
    parser.add_argument('--window-size', type=int, default=7)
    parser.add_argument('--class-weights', type=float, nargs='+', default=None,
                        help='Class weights for CrossEntropyLoss, e.g. --class-weights 1.0 9.0 '
                             'to upweight the minority class (true). '
                             'Order matches class index (false=0, true=1).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility. Not set by default. '
                             'Use e.g. --seed 42 to enable.')

    opt = parser.parse_args()
    main(opt)
