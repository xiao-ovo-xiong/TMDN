"""
Twin Network Implementation
TwinFusionModule + TwinNetwork

将 MultiModalConvNeXt 和 MultiModalSwinTransformer 组合为双胞胎网络，
支持 logits_gating 和 feature_gating 两种融合策略。
"""

import os
import sys

import torch
import torch.nn as nn

from multimodal_convnext import MultiModalConvNeXt

# 将 swin_transformer 目录加入路径
_root = os.path.dirname(os.path.abspath(__file__))
_swin_dir = os.path.join(_root, 'swin_transformer')
if _swin_dir not in sys.path:
    sys.path.insert(0, _swin_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from swin_transformer.multimodal_swin import MultiModalSwinTransformer


class TwinFusionModule(nn.Module):
    """
    双胞胎网络融合模块

    融合 MultiModalConvNeXt 和 MultiModalSwinTransformer 两个子网络的输出。
    支持两种策略：
      - logits_gating: 对两个子网络的 logits 做动态门控加权融合
      - prob_gating: 对两个子网络的 softmax 概率做门控，融合仍用原始 logits
      - feature_gating: 对两个子网络的特征向量做逐元素门控融合，再经共享分类头输出

    Args:
        fusion_type (str): 'logits_gating' 或 'feature_gating'
        feature_dim_convnext (int): ConvNeXt 特征维度
        feature_dim_swin (int): Swin 特征维度
        num_classes (int): 分类类别数
    """

    _VALID_FUSION_TYPES = ('logits_gating', 'prob_gating', 'feature_gating')

    def __init__(
        self,
        fusion_type: str,
        feature_dim_convnext: int,
        feature_dim_swin: int,
        num_classes: int,
    ):
        super().__init__()

        if fusion_type not in self._VALID_FUSION_TYPES:
            raise ValueError(
                f"Invalid fusion_type: '{fusion_type}'. "
                f"Must be one of {list(self._VALID_FUSION_TYPES)}"
            )

        self.fusion_type = fusion_type
        self.feature_dim_convnext = feature_dim_convnext
        self.feature_dim_swin = feature_dim_swin
        self.num_classes = num_classes

        if fusion_type == 'logits_gating':
            # 门控输入：两个分支的原始 logits 拼接
            self.gate = nn.Linear(2 * num_classes, 2)
            nn.init.zeros_(self.gate.bias)
            nn.init.normal_(self.gate.weight, std=1e-4)

        elif fusion_type == 'prob_gating':
            # 门控输入：两个分支 softmax 后的概率拼接，信号更稳定
            self.gate = nn.Linear(2 * num_classes, 2)
            nn.init.zeros_(self.gate.bias)
            nn.init.normal_(self.gate.weight, std=1e-4)

        else:  # feature_gating
            D = max(feature_dim_convnext, feature_dim_swin)
            self.fused_dim = D

            # 维度对齐投影层（若维度不一致）
            self.proj_convnext = (
                nn.Linear(feature_dim_convnext, D)
                if feature_dim_convnext != D else nn.Identity()
            )
            self.proj_swin = (
                nn.Linear(feature_dim_swin, D)
                if feature_dim_swin != D else nn.Identity()
            )

            # 门控网络：输入为两个投影特征的拼接，输出逐元素门控掩码
            self.gate = nn.Linear(2 * D, D)
            # 初始化：权重接近零，bias=0，使 sigmoid 输出趋近 0.5
            nn.init.zeros_(self.gate.bias)
            nn.init.normal_(self.gate.weight, std=1e-4)

            # 共享分类头
            self.shared_head = nn.Linear(D, num_classes)

    def forward(
        self,
        logits_convnext: torch.Tensor,
        logits_swin: torch.Tensor,
        feat_convnext: torch.Tensor = None,
        feat_swin: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits_convnext: [B, num_classes]
            logits_swin:     [B, num_classes]
            feat_convnext:   [B, feature_dim_convnext]（feature_gating 时使用）
            feat_swin:       [B, feature_dim_swin]（feature_gating 时使用）

        Returns:
            融合后的 logits [B, num_classes]
        """
        if self.fusion_type == 'logits_gating':
            gate_input = torch.cat([logits_convnext, logits_swin], dim=-1)
            gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
            return gate_weights[:, 0:1] * logits_convnext + gate_weights[:, 1:2] * logits_swin

        elif self.fusion_type == 'prob_gating':
            gate_input = torch.cat([
                torch.softmax(logits_convnext, dim=-1),
                torch.softmax(logits_swin, dim=-1)
            ], dim=-1)
            gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
            return gate_weights[:, 0:1] * logits_convnext + gate_weights[:, 1:2] * logits_swin

        else:  # feature_gating
            fc = self.proj_convnext(feat_convnext)   # [B, D]
            fs = self.proj_swin(feat_swin)            # [B, D]
            gate_input = torch.cat([fc, fs], dim=-1)  # [B, 2D]
            gate = torch.sigmoid(self.gate(gate_input))  # [B, D]
            fused = gate * fc + (1.0 - gate) * fs        # [B, D]
            return self.shared_head(fused)               # [B, num_classes]


class TwinNetwork(nn.Module):
    """
    双胞胎网络

    由 MultiModalConvNeXt 和 MultiModalSwinTransformer 两个子网络组成，
    通过 TwinFusionModule 融合两个子网络的输出。

    Args:
        num_classes (int): 分类类别数
        convnext_depths (list): ConvNeXt 各阶段深度
        convnext_dims (list): ConvNeXt 各阶段通道数
        convnext_fusion_type (str): ConvNeXt 内部融合类型
        swin_embed_dim (int): Swin embed_dim
        swin_depths (tuple): Swin 各阶段深度
        swin_num_heads (tuple): Swin 各阶段注意力头数
        swin_window_size (int): Swin 窗口大小
        swin_fusion_type (str): Swin 内部融合类型
        twin_fusion_type (str): 双胞胎融合类型，'logits_gating'、'prob_gating' 或 'feature_gating'
    """

    def __init__(
        self,
        num_classes: int,
        convnext_depths: list = None,
        convnext_dims: list = None,
        convnext_fusion_type: str = 'weighted_sum',
        swin_embed_dim: int = 96,
        swin_depths: tuple = (2, 2, 6, 2),
        swin_num_heads: tuple = (3, 6, 12, 24),
        swin_window_size: int = 7,
        swin_fusion_type: str = 'weighted_sum',
        twin_fusion_type: str = 'logits_gating',
    ):
        super().__init__()

        if convnext_depths is None:
            convnext_depths = [3, 3, 27, 3]
        if convnext_dims is None:
            convnext_dims = [96, 192, 384, 768]

        self.twin_fusion_type = twin_fusion_type

        # ConvNeXt 子网络
        self.convnext = MultiModalConvNeXt(
            num_classes=num_classes,
            depths=convnext_depths,
            dims=convnext_dims,
            fusion_type=convnext_fusion_type,
        )

        # Swin 子网络
        self.swin = MultiModalSwinTransformer(
            num_classes=num_classes,
            embed_dim=swin_embed_dim,
            depths=swin_depths,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            fusion_type=swin_fusion_type,
        )

        feature_dim_convnext = convnext_dims[-1]
        num_layers = len(swin_depths)
        feature_dim_swin = int(swin_embed_dim * 2 ** (num_layers - 1))

        # 双胞胎融合模块
        self.twin_fusion = TwinFusionModule(
            fusion_type=twin_fusion_type,
            feature_dim_convnext=feature_dim_convnext,
            feature_dim_swin=feature_dim_swin,
            num_classes=num_classes,
        )

    def forward(
        self,
        original_c: torch.Tensor,
        wavelet_c: torch.Tensor,
        fourier_c: torch.Tensor,
        original_s: torch.Tensor,
        wavelet_s: torch.Tensor,
        fourier_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            original_c/wavelet_c/fourier_c: ConvNeXt 的三模态输入 [B, 3, H, W]
            original_s/wavelet_s/fourier_s: Swin 的三模态输入 [B, 3, H, W]

        Returns:
            融合后的 logits [B, num_classes]
        """
        if self.twin_fusion_type in ('logits_gating', 'prob_gating'):
            logits_c = self.convnext(original_c, wavelet_c, fourier_c)
            logits_s = self.swin(original_s, wavelet_s, fourier_s)
            return self.twin_fusion(logits_c, logits_s)
        else:  # feature_gating
            feat_c = self.convnext.forward_multimodal_features(original_c, wavelet_c, fourier_c)
            feat_s = self.swin.forward_multimodal_features(original_s, wavelet_s, fourier_s)
            # logits 参数在 feature_gating 模式下不使用，传 None 占位
            return self.twin_fusion(None, None, feat_c, feat_s)

    def load_pretrained_convnext(self, weights_path: str) -> None:
        """为 ConvNeXt 子网络加载预训练权重"""
        if not weights_path or not os.path.exists(weights_path):
            print(f"警告: ConvNeXt 预训练权重路径无效或未指定: {weights_path}，使用随机初始化")
            return
        self.convnext.load_pretrained(weights_path, strict=False)

    def load_pretrained_swin(self, weights_path: str) -> None:
        """为 Swin 子网络加载预训练权重"""
        if not weights_path or not os.path.exists(weights_path):
            print(f"警告: Swin 预训练权重路径无效或未指定: {weights_path}，使用随机初始化")
            return
        self.swin.load_pretrained(weights_path, strict=False)
