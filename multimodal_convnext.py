"""
Multi-Modal ConvNeXt Implementation
支持多模态输入（原始图像、小波变换图像、傅里叶变换图像）的ConvNeXt网络
"""

import os
import torch
import torch.nn as nn
from model import ConvNeXt
from fusion_module import FusionModule


class MultiModalConvNeXt(nn.Module):
    """
    多模态ConvNeXt网络
    
    接收三种模态的图像输入（原始图像、小波变换图像、傅里叶变换图像），
    通过独立的特征提取器提取特征，然后融合特征进行分类。
    
    Args:
        num_classes (int): 分类类别数
        depths (list): ConvNeXt各阶段深度，默认[3, 3, 27, 3]
        dims (list): ConvNeXt各阶段通道数，默认[96, 192, 384, 768]
        fusion_type (str): 特征融合类型，默认'weighted_sum'
        shared_weights (bool): 三个特征提取器是否共享权重，默认False
        drop_path_rate (float): DropPath率，默认0.0
    """
    
    def __init__(
        self,
        num_classes: int,
        depths: list = None,
        dims: list = None,
        fusion_type: str = 'weighted_sum',
        shared_weights: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        
        if depths is None:
            depths = [3, 3, 27, 3]
        if dims is None:
            dims = [96, 192, 384, 768]
        
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.fusion_type = fusion_type
        self.shared_weights = shared_weights
        self.drop_path_rate = drop_path_rate
        
        self.feature_dim = dims[-1]
        
        if shared_weights:
            base_extractor = self._create_feature_extractor(depths, dims, drop_path_rate)
            self.extractor_original = base_extractor
            self.extractor_wavelet = base_extractor
            self.extractor_fourier = base_extractor
        else:
            self.extractor_original = self._create_feature_extractor(depths, dims, drop_path_rate)
            self.extractor_wavelet = self._create_feature_extractor(depths, dims, drop_path_rate)
            self.extractor_fourier = self._create_feature_extractor(depths, dims, drop_path_rate)
        
        self.fusion_module = FusionModule(
            feature_dim=self.feature_dim,
            fusion_type=fusion_type,
            learnable=True
        )
        
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def _create_feature_extractor(self, depths: list, dims: list, drop_path_rate: float) -> ConvNeXt:
        """
        创建ConvNeXt特征提取器（移除分类头）
        
        Args:
            depths: ConvNeXt各阶段深度
            dims: ConvNeXt各阶段通道数
            drop_path_rate: DropPath率
            
        Returns:
            ConvNeXt模型实例（仅用于特征提取）
        """
        extractor = ConvNeXt(
            in_chans=3,
            num_classes=1000,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate
        )
        
        return extractor

    def forward(
        self,
        original: torch.Tensor,
        wavelet: torch.Tensor,
        fourier: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播方法
        
        Args:
            original: 原始图像张量 [batch_size, 3, H, W]
            wavelet: 小波变换图像张量 [batch_size, 3, H, W]
            fourier: 傅里叶变换图像张量 [batch_size, 3, H, W]
            
        Returns:
            分类logits [batch_size, num_classes]
            
        Raises:
            ValueError: 如果三个输入的形状不一致
        """
        if original.shape != wavelet.shape or original.shape != fourier.shape:
            raise ValueError(
                f"Input shape mismatch: original={original.shape}, "
                f"wavelet={wavelet.shape}, fourier={fourier.shape}"
            )
        
        feat_original = self.extractor_original.forward_features(original)
        feat_wavelet = self.extractor_wavelet.forward_features(wavelet)
        feat_fourier = self.extractor_fourier.forward_features(fourier)
        
        fused_features = self.fusion_module(feat_original, feat_wavelet, feat_fourier)
        
        logits = self.classifier(fused_features)
        
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
            original: 原始图像张量 [batch_size, 3, H, W]
            wavelet: 小波变换图像张量 [batch_size, 3, H, W]
            fourier: 傅里叶变换图像张量 [batch_size, 3, H, W]

        Returns:
            融合特征向量 [batch_size, feature_dim]
        """
        if original.shape != wavelet.shape or original.shape != fourier.shape:
            raise ValueError(
                f"Input shape mismatch: original={original.shape}, "
                f"wavelet={wavelet.shape}, fourier={fourier.shape}"
            )
        feat_original = self.extractor_original.forward_features(original)
        feat_wavelet = self.extractor_wavelet.forward_features(wavelet)
        feat_fourier = self.extractor_fourier.forward_features(fourier)
        return self.fusion_module(feat_original, feat_wavelet, feat_fourier)

    def load_pretrained(
        self,
        weights_path: str,
        strict: bool = False
    ) -> None:
        """
        加载预训练权重到特征提取器
        
        仅初始化特征提取器部分，保持FusionModule和分类头随机初始化。
        支持共享权重和独立权重两种模式。
        
        Args:
            weights_path: 预训练权重文件路径（.pth或.pt文件）
            strict: 是否严格匹配所有键。如果为True，权重字典必须完全匹配模型结构；
                   如果为False，允许部分匹配（推荐用于迁移学习）
                   
        Raises:
            FileNotFoundError: 如果权重文件不存在
            RuntimeError: 如果权重加载失败或格式不匹配
            
        Example:
            >>> model = MultiModalConvNeXt(num_classes=5)
            >>> model.load_pretrained('convnext_tiny_1k_224_ema.pth')
            >>> # 现在特征提取器已初始化，可以开始训练
        """
        # 1. 验证权重文件存在性
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"预训练权重文件不存在: {weights_path}\n"
                f"请检查文件路径是否正确。"
            )
        
        # 2. 加载权重文件
        try:
            print(f"加载预训练权重: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(
                f"无法加载权重文件 {weights_path}: {str(e)}\n"
                f"请确保文件格式正确且未损坏。"
            ) from e
        
        # 3. 提取state_dict
        # 权重文件可能直接是state_dict，也可能包含在'model'或'state_dict'键中
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            raise RuntimeError(
                f"权重文件格式不正确: 期望dict类型，实际为{type(checkpoint)}"
            )
        
        try:
            if self.shared_weights:
                missing_keys, unexpected_keys = self.extractor_original.load_state_dict(
                    state_dict, strict=strict
                )
                print(f"✓ ConvNeXt 预训练权重加载完成（共享模式）")
            else:
                missing_keys_1, _ = self.extractor_original.load_state_dict(state_dict, strict=strict)
                missing_keys_2, _ = self.extractor_wavelet.load_state_dict(state_dict, strict=strict)
                missing_keys_3, _ = self.extractor_fourier.load_state_dict(state_dict, strict=strict)
                print(f"✓ ConvNeXt 预训练权重加载完成（3个独立提取器）")
            print(f"  FusionModule 和 classifier 保持随机初始化")
        except Exception as e:
            raise RuntimeError(
                f"加载权重到特征提取器时失败: {str(e)}"
            ) from e
