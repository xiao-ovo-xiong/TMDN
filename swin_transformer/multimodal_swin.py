"""
Multi-Modal Swin Transformer Implementation
支持多模态输入（原始图像、小波变换图像、傅里叶变换图像）的 Swin Transformer 网络
与根目录 multimodal_convnext.py 结构对称
"""

import os
import sys

# 将父目录加入 sys.path，以便导入根目录的 fusion_module.py
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import torch
import torch.nn as nn

from swin_transformer.model import SwinTransformer
from fusion_module import FusionModule


class MultiModalSwinTransformer(nn.Module):
    """
    多模态 Swin Transformer 网络

    接收三种模态的图像输入（原始图像、小波变换图像、傅里叶变换图像），
    通过独立的 SwinTransformer encoder 提取特征，然后通过 FusionModule 融合特征进行分类。

    Args:
        num_classes (int): 分类类别数
        embed_dim (int): Patch embedding 维度，默认 96
        depths (tuple): 各阶段 Transformer block 数量，默认 (2, 2, 6, 2)
        num_heads (tuple): 各阶段注意力头数，默认 (3, 6, 12, 24)
        window_size (int): 窗口大小，默认 7
        fusion_type (str): 特征融合类型，默认 'weighted_sum'
        shared_weights (bool): 三个 encoder 是否共享权重，默认 False
        drop_path_rate (float): Stochastic depth rate，默认 0.1
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: int = 7,
        fusion_type: str = 'weighted_sum',
        shared_weights: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.fusion_type = fusion_type
        self.shared_weights = shared_weights
        self.drop_path_rate = drop_path_rate

        # num_features = embed_dim * 2^(num_layers - 1)
        num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (num_layers - 1))

        # 创建 encoder（num_classes=0 时 head=Identity，forward() 直接返回特征向量）
        if shared_weights:
            base_encoder = self._create_encoder()
            self.encoder_original = base_encoder
            self.encoder_wavelet = base_encoder
            self.encoder_fourier = base_encoder
        else:
            self.encoder_original = self._create_encoder()
            self.encoder_wavelet = self._create_encoder()
            self.encoder_fourier = self._create_encoder()

        # 特征融合模块
        self.fusion_module = FusionModule(
            feature_dim=self.num_features,
            fusion_type=fusion_type,
            learnable=True,
        )

        # 独立线性分类头
        self.classifier = nn.Linear(self.num_features, num_classes)

    def _create_encoder(self) -> SwinTransformer:
        """创建 SwinTransformer encoder（num_classes=0，head 为 Identity）"""
        return SwinTransformer(
            patch_size=4,
            in_chans=3,
            num_classes=0,  # head = nn.Identity()，forward() 直接返回特征向量
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            drop_path_rate=self.drop_path_rate,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        对单个模态输入提取特征向量

        Args:
            x: 输入张量 [B, 3, H, W]

        Returns:
            特征向量 [B, num_features]
        """
        # num_classes=0 时，SwinTransformer.forward() 经过 head=Identity 直接返回特征向量
        return self.encoder_original(x)

    def forward(
        self,
        original: torch.Tensor,
        wavelet: torch.Tensor,
        fourier: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            original: 原始图像张量 [B, 3, H, W]
            wavelet: 小波变换图像张量 [B, 3, H, W]
            fourier: 傅里叶变换图像张量 [B, 3, H, W]

        Returns:
            分类 logits [B, num_classes]

        Raises:
            ValueError: 如果三个输入的形状不一致
        """
        if original.shape != wavelet.shape or original.shape != fourier.shape:
            raise ValueError(
                f"Input shape mismatch: original={original.shape}, "
                f"wavelet={wavelet.shape}, fourier={fourier.shape}"
            )

        feat_original = self.encoder_original(original)
        feat_wavelet = self.encoder_wavelet(wavelet)
        feat_fourier = self.encoder_fourier(fourier)

        fused = self.fusion_module(feat_original, feat_wavelet, feat_fourier)
        logits = self.classifier(fused)
        return logits

    def forward_multimodal_features(
        self,
        original: torch.Tensor,
        wavelet: torch.Tensor,
        fourier: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回融合后的特征向量（分类头之前），供 TwinNetwork 的 feature_gating 模式使用

        Args:
            original: 原始图像张量 [B, 3, H, W]
            wavelet: 小波变换图像张量 [B, 3, H, W]
            fourier: 傅里叶变换图像张量 [B, 3, H, W]

        Returns:
            融合特征向量 [B, num_features]

        Raises:
            ValueError: 如果三个输入的形状不一致
        """
        if original.shape != wavelet.shape or original.shape != fourier.shape:
            raise ValueError(
                f"Input shape mismatch: original={original.shape}, "
                f"wavelet={wavelet.shape}, fourier={fourier.shape}"
            )

        feat_original = self.encoder_original(original)
        feat_wavelet = self.encoder_wavelet(wavelet)
        feat_fourier = self.encoder_fourier(fourier)

        fused = self.fusion_module(feat_original, feat_wavelet, feat_fourier)
        return fused

    def load_pretrained(self, weights_path: str, strict: bool = False) -> None:
        """
        加载预训练权重到三个独立 encoder，FusionModule 和 classifier 保持随机初始化

        Args:
            weights_path: 预训练权重文件路径
            strict: 是否严格匹配所有键，默认 False

        Raises:
            FileNotFoundError: 如果权重文件不存在
            RuntimeError: 如果权重加载失败
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"预训练权重文件不存在: {weights_path}")

        try:
            print(f"加载预训练权重: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"无法加载权重文件 {weights_path}: {str(e)}") from e

        # 提取 state_dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            raise RuntimeError(f"权重文件格式不正确: 期望 dict 类型，实际为 {type(checkpoint)}")

        # 构建 encoder 用的 state_dict：
        # 支持两种格式：
        #   1. 裸 encoder 权重（如官方预训练权重，键名如 patch_embed.proj.weight）
        #   2. MultiModalSwinTransformer 完整权重（键名如 encoder_original.patch_embed.proj.weight）
        # 统一转换为裸 encoder 格式（去掉 encoder_original. 前缀），并过滤 head/fusion/classifier
        encoder_prefix = 'encoder_original.'
        if any(k.startswith(encoder_prefix) for k in state_dict.keys()):
            # 格式2：从完整模型权重中提取 encoder_original 部分
            encoder_state_dict = {
                k[len(encoder_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(encoder_prefix)
            }
        else:
            # 格式1：裸 encoder 权重，过滤掉 head
            encoder_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith('head.')
            }

        def _load_to_encoder(encoder, name):
            missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=strict)
            if missing:
                print(f"  警告 [{name}]: 缺失键 {len(missing)} 个")
            # 过滤掉 attn_mask（register_buffer，非可学习参数，可安全忽略）
            real_unexpected = [k for k in unexpected if 'attn_mask' not in k]
            if real_unexpected:
                print(f"  警告 [{name}]: 意外键 {len(real_unexpected)} 个")

        if self.shared_weights:
            _load_to_encoder(self.encoder_original, "共享 encoder")
            print(f"✓ Swin 预训练权重加载完成（共享模式）")
        else:
            _load_to_encoder(self.encoder_original, "encoder_original")
            _load_to_encoder(self.encoder_wavelet, "encoder_wavelet")
            _load_to_encoder(self.encoder_fourier, "encoder_fourier")
            print(f"✓ Swin 预训练权重加载完成（3个独立编码器）")

        print(f"  FusionModule 和 classifier 保持随机初始化")
